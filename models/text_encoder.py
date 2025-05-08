# -*- coding: utf-8 -*-
"""
@Project : SegChange-R1
@FileName: text_encoder.py
@Time    : 2025/4/17 下午3:40
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : TextEncoder
@Usage   :
"""
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertTokenizer, BertModel


class TextEncoderLLM(nn.Module):
    def __init__(self, model_name="microsoft/phi-1_5", device='cuda', freeze_text_encoder=True):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # 设置pad_token，用于填充
        self.tokenizer.pad_token = self.tokenizer.eos_token
        max_memory = {0: "10GB"}
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float32,
            max_memory=max_memory).to(self.device)  # , local_files_only=True

        # 冻结模型
        if freeze_text_encoder:
            for p in self.llm.parameters():
                p.requires_grad = False

    def forward(self, prompts):
        """
        prompts: list of str
        """
        # 对每个prompt单独编码，并收集输入ids
        input_ids_list = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids_list.append(inputs["input_ids"])

        # 找到最长的输入长度
        max_length = max([input_ids.shape[1] for input_ids in input_ids_list])

        # 对每个输入进行填充，使其长度一致
        padded_input_ids = []
        for input_ids in input_ids_list:
            # 计算需要填充的长度
            pad_length = max_length - input_ids.shape[1]
            # 填充
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1))

        # 将填充后的输入合并成一个批次
        batch_input_ids = torch.cat(padded_input_ids, dim=0).to(self.device)

        # 获取注意力掩码
        attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).to(self.device)

        # 通过模型获取隐藏状态
        outputs = self.llm.model(input_ids=batch_input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # last hidden state: [B, seq_len, hidden_size]
        return outputs.hidden_states[-1], batch_input_ids  # (desc_embs, tokenized_input)

    def to(self, device):
        self.device = device
        self.llm = self.llm.to(device)
        return self


class TextEncoderBert(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device='cuda', freeze_text_encoder=True):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name).to(self.device)

        # 冻结模型
        if freeze_text_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, prompts):
        """
        prompts: list of str
        """
        # 对每个prompt单独编码，并收集输入ids
        input_ids_list = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids_list.append(inputs["input_ids"])

        # 找到最长的输入长度
        max_length = max([input_ids.shape[1] for input_ids in input_ids_list])

        # 对每个输入进行填充，使其长度一致
        padded_input_ids = []
        for input_ids in input_ids_list:
            # 计算需要填充的长度
            pad_length = max_length - input_ids.shape[1]
            # 填充
            padded_input_ids.append(
                torch.cat([input_ids, torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=1))

        # 将填充后的输入合并成一个批次
        batch_input_ids = torch.cat(padded_input_ids, dim=0).to(self.device)
        attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).to(self.device)

        # 通过模型获取隐藏状态
        outputs = self.bert(batch_input_ids, attention_mask=attention_mask)
        # last hidden state: [B, seq_len, hidden_size]
        return outputs.last_hidden_state, batch_input_ids  # 返回CLS token的特征

    def to(self, device):
        self.device = device
        self.bert = self.bert.to(device)
        return self


# 测试
if __name__ == '__main__':
    model = TextEncoderBert()
    # model = TextEncoderLLM(model_name="./local_models/microsoft/phi-1_5")
    prompts = ["This is a test prompt.", "This is another test prompt."]
    desc_embs, tokenized_input = model(prompts)
    print(desc_embs.shape)

    from thop import profile
    flops, params = profile(model, inputs=(prompts,))
    print(f"FLOPs: {flops / 1e9} G, Params: {params / 1e6} M")
