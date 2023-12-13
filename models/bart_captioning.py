#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


# PANNs - BART audio captioning model

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from models.audio_encoder_config import AudioEncoderConfig
from models.audio_encoder import AudioEncoderModel
from IPython import embed
import torch.nn.functional as F



class BartCaptionModel(nn.Module):

    def __init__(self, config):
        super(BartCaptionModel, self).__init__()

        self.config = config

        # encoder
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],
                                            audio_args=config["audio_args"])
        self.encoder = AudioEncoderModel(encoder_config)

        # bart decoder
        decoder_name = config["text_decoder_args"]["name"]
        
        self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
        self.tokenizer = BartTokenizer.from_pretrained(decoder_name)

        
        self.enc_to_dec_proj = nn.Linear(encoder_config.hidden_size, self.decoder.config.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids



    def forward_encoder(self, audios, keywords=None):
        #print(audios.shape)
        #print(len(keywords))
        max_len= 10
        outputs = self.encoder(audios)

        if keywords is not None:
            keywords_tokens = self.tokenizer(keywords,
                                            padding='longest',
                                            truncation=True,
                                            max_length=max_len,
                                            return_tensors="pt")
            #print('tokenized')
            
            keywords_input_ids = keywords_tokens["input_ids"].to(self.device)
            #print('input_id')
            keywords_attention_mask = keywords_tokens["attention_mask"].to(self.device)
            #print('attention mask')

            keywords_embedding = self.decoder(input_ids = keywords_input_ids,
                                            attention_mask = keywords_attention_mask,
                                            output_hidden_states=True)
                                            #   return_dict=True)
            

            emb_len = keywords_embedding.encoder_last_hidden_state.shape[-2]
            # p = outputs.last_hidden_state[:, -emb_len:, :]
            # print(p.shape)

            # print(outputs.last_hidden_state[:, -emb_len:, :])
            # print('after---------------------------------------------------')
            
            outputs.last_hidden_state[:, -emb_len:, :] = keywords_embedding.encoder_last_hidden_state
        # print(outputs.last_hidden_state[:, -emb_len:, :])

        # if(emb_len < max_len):
        #     pad_length = max_len-emb_len
        #     keywords_embedding.encoder_last_hidden_state = F.pad(keywords_embedding.encoder_last_hidden_state, [0, pad_length], "constant", 0.0)

        
        # audio_keyword = torch.concat((outputs.last_hidden_state, keywords_embedding.encoder_last_hidden_state), dim=1)
        
        # print(audio_keyword.shape)
        
        # outputs = self.enc_to_dec_proj(audio_keyword)
        outputs = self.enc_to_dec_proj(outputs.last_hidden_state)
        
        return outputs
    

    def forward_encoder_gen(self, audios):
        
        outputs = self.encoder(audios)

        outputs = self.enc_to_dec_proj(outputs.last_hidden_state)
    
        return outputs


    def forward_decoder(self, text, encoder_outputs):
        
        # Feed the audio embeddings to the bart encoder
        encoder_outputs = self.decoder.model.encoder(
            input_ids=None,
            inputs_embeds=encoder_outputs,
            return_dict=True
        )["last_hidden_state"]
        print('encode outputs')
        text = self.tokenizer(text,
                              padding='longest',
                              truncation=True,
                              max_length=30,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_input_ids = self.shift_tokens_right(
            decoder_targets, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
        )


        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask,
            inputs_embeds=None,
            labels=None,
            encoder_outputs=(encoder_outputs,),
            return_dict=True
        )

        lm_logits = decoder_outputs["logits"]
        loss = self.loss_fct(lm_logits.view(-1, self.tokenizer.vocab_size), decoder_targets.view(-1))
       
        return loss

    def forward(self, audio, text, keywords=None):

        audio_embeds = self.forward_encoder(audio, keywords)
        loss = self.forward_decoder(text, audio_embeds)
        return loss

    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=3,
                 max_length=30,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):

        audio_embs = self.forward_encoder(samples)
        # audio_embs = self.forward_encoder_gen(samples)


        # Encoder pass
        encoder_outputs = self.decoder.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=audio_embs,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True)

        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)
        input_ids[:, 0] = self.decoder.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0), 1)).long().to(self.device)


        if use_nucleus_sampling:
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=1.1)
        else:
            outputs = self.decoder.generate(input_ids=None,
                                            attention_mask=None,
                                            decoder_input_ids=input_ids,
                                            decoder_attention_mask=decoder_attention_mask,
                                            encoder_outputs=encoder_outputs,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            inputs_embeds=None,
                                            decoder_inputs_embeds=None,
                                            use_cache=None,
                                            output_attentions=None,
                                            output_hidden_states=None,
                                            max_length=max_length,
                                            min_length=min_length,
                                            num_beams=num_beams,
                                            repetition_penalty=repetition_penalty)

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return captions
