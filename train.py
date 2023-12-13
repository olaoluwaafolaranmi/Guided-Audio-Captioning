#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import time
import torch
import argparse
import platform
import numpy as np
from tqdm import tqdm

from loguru import logger
import ruamel.yaml as yaml
from pprint import PrettyPrinter
from fense.evaluator import Evaluator
from data_handling.datamodule import AudioCaptionDataModule
from models.bart_captioning import BartCaptionModel

from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import setup_seed, set_logger, AverageMeter, decode_output
from eval_metrics import evaluate_metrics
from IPython import embed
import os



os.environ["TOKENIZERS_PARALLELISM"] = "false"



def train(model, dataloader, optimizer, scheduler, device, epoch):
    
    model.train()

    epoch_loss = AverageMeter()
    start_time = time.time()

    for batch_id, (audio, text, audio_names, _, keywords) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
      
        step = len(dataloader) * (epoch - 1) + batch_id
        
        if scheduler is not None:
            scheduler(step)
       
        audio = audio.to(device, non_blocking=True)
       
        loss = model(audio, text, keywords)
        
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())
     

    elapsed_time = time.time() - start_time

    print({"loss": epoch_loss.avg,
               "epoch": epoch})
    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


@torch.no_grad()
def validate(data_loader, model, device, log_dir, epoch, beam_size):
    val_logger = logger.bind(indent=1)
    model.eval()
    with torch.no_grad():
        y_hat_all = []
        ref_captions_dict = []
        file_names_all = []
        start_time = time.time()

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, caption_dict, audio_names, audio_ids = batch_data
            # move data to GPU
            audios = audios.to(device)
            # Where the caption is generated
            output = model.generate(samples=audios,
                                    num_beams=beam_size)

            y_hat_all.extend(output)
            ref_captions_dict.extend(caption_dict)
            file_names_all.extend(audio_names)

        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_dict, file_names_all,
                                                   log_dir, epoch, beam_size=beam_size)
        metrics = evaluate_metrics(captions_pred, captions_gt)

        bleu1 = metrics['bleu_1']['score']
        bleu4 = metrics['bleu_4']['score']
        rouge = metrics['rouge_l']['score']
        meteor = metrics['meteor']['score']
        cider = metrics['cider']['score']
        spice = metrics['spice']['score']
        spider = metrics['spider']['score']
        
        eval_time = time.time() - start_time

        val_logger.info(f'Bleu_1: {bleu1:7.4f}\n Bleu_4: {bleu4:7.4f}\n Rouge_l: {rouge:7.4f}\n Meteor: {meteor:7.4f}\n Cider: {cider:7.4f}\n Spice: {spice:7.4f} ')
        val_logger.info(f'Spider score : {spider:7.4f}, eval time: {eval_time:.1f}')

        # # Write outputs to disk
        # with open('outputs/generated_captions_htsatbart_clotho.txt', 'w') as f:
        #     for i_file in range(len(captions_pred)):
        #         f.write('----- File {} -----\n'.format(i_file))
        #         f.write('GT:   '+'\n')
        #         for i_gt in list(captions_gt[i_file].keys())[1:]:
        #             f.write('      '+captions_gt[i_file][i_gt]+'\n')
        #         f.write('Pred: '+captions_pred[i_file]['caption_predicted']+'\n')

        ## Calculate SPIDER + fluency error 
        fense_eval = Evaluator(device='cuda' if torch.cuda.is_available() else 'cpu', sbert_model=None)
        predictions = []
        for row in captions_pred:
            predictions.append(row['caption_predicted'])
        fl_err = fense_eval.detect_error_sents(predictions, batch_size=32)
    
        # Add error-penalized SPIDEr as SPIDEr-FL
        spider_scores = metrics['spider']['scores']
        spider_fl_scores = {k: spider_scores[k]-0.9*err*spider_scores[k] for k,err in zip(sorted(spider_scores.keys()), fl_err)} # Divide score by 10 if an error is found
        metrics['SPIDEr_fl'] = {'score':np.mean([v for k,v in spider_fl_scores.items()]), 'scores': spider_fl_scores}
        print('SPIDEr-FL: {:.3f}'.format(metrics['SPIDEr_fl']['score']))

        # Write metrics to disk
        #write_json(metrics, Path('outputs/metrics_coco_htsatbart_clotho.json'))

        return metrics




def main():
    parser = argparse.ArgumentParser(description='Settings.')
    parser.add_argument('-n', '--exp_name', default='htsat_test_guide', type=str,
                        help='Name of the experiment.')
    parser.add_argument('-c', '--config', default='settings/settings.yaml', type=str,
                        help='Name of the setting file.')
    parser.add_argument('-l', '--lr', default=1e-04, type=float,
                        help='Learning rate.')
    parser.add_argument('-s', '--seed', default=20, type=int,
                        help='Training seed.')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["exp_name"] = args.exp_name
    config["seed"] = args.seed
    config["optim_args"]["lr"] = args.lr

    setup_seed(config["seed"])

    exp_name = config["exp_name"]

    folder_name = '{}_{}_batch_{}_seed_{}'.format(exp_name,
                                                     'framewise_tags',
                                                     config["data_args"]["batch_size"],
                                                     config["seed"])

    model_output_dir, log_output_dir = set_logger(folder_name)

    main_logger = logger.bind(indent=1)

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    # data loading
    datamodule = AudioCaptionDataModule(config, config["data_args"]["dataset"])
    train_loader = datamodule.train_dataloader(is_distributed=False)
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()


    model = BartCaptionModel(config)

    model = model.to(device)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')


    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')

    if config["pretrain"]:
        pretrain_checkpoint = torch.load(config["pretrain_path"])
        model.load_state_dict(pretrain_checkpoint["model"])
        main_logger.info(f"Loaded weights from {config['pretrain_path']}")

    # set up optimizer and loss
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              weight_decay=config["optim_args"]["weight_decay"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    # scheduler = None
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(train_loader),
                          steps=len(train_loader) * config["training"]["epochs"])


    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    # training loop
    loss_stats = []
    spiders = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        train_statics = train(model, train_loader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        # validation loop, validation after each epoch
        main_logger.info("Validating...")
        
        metrics = validate(val_loader,
                            model,
                            device=device,
                            log_dir=log_output_dir,
                            epoch=epoch,
                            beam_size=3)
        spider = metrics["spider"]["score"]

        spiders.append(spider)

        if spider >= max(spiders):
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "beam_size": 4,
                "epoch": epoch,
                "config": config,
            }, str(model_output_dir) + '/best_model.pt'.format(epoch))

    # Training done, evaluate on evaluation set
    main_logger.info('Training done. Start evaluating.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pt')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint occurred in {best_epoch} th epoch.')
    
    spider = validate(test_loader, model,
                        device=device,
                        log_dir=log_output_dir,
                        epoch=0,
                        beam_size=4,
                        )['spider']['score']

    main_logger.info('Evaluation done.')



if __name__ == '__main__':
    main()
