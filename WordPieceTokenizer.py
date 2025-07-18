import torch
import re
import unicodedata

class WordPieceTokenizer:
    def __init__(self, vocab_file_path, do_lower_case=False, strip_accents=False, clean_text=True):
        self.vocab = self._load_vocab(vocab_file_path)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"

        self.cls_token_id = self.vocab.get(self.cls_token, -1)
        self.sep_token_id = self.vocab.get(self.sep_token, -1)
        self.unk_token_id = self.vocab.get(self.unk_token, -1)
        self.pad_token_id = self.vocab.get(self.pad_token, -1)
        self.mask_token_id = self.vocab.get(self.mask_token, -1)

        self.do_lower_case = do_lower_case
        self.strip_accents = strip_accents
        self.clean_text = clean_text

        if self.unk_token_id == -1:
            print(f"'{self.unk_token}' 토큰이 vocab에 없습니다. UNK 토큰 처리에 문제가 발생할 수 있습니다.")
        if self.cls_token_id == -1:
            print(f"'{self.cls_token}' 토큰이 vocab에 없습니다.")
        if self.sep_token_id == -1:
            print(f"'{self.sep_token}' 토큰이 vocab에 없습니다.")
        if self.pad_token_id == -1:
            print(f"'{self.pad_token}' 토큰이 vocab에 없습니다.")
        if self.mask_token_id == -1:
            print(f"'{self.mask_token}' 토큰이 vocab에 없습니다.")

    def _load_vocab(self, vocab_file_path):
        vocab = {}
        with open(vocab_file_path, "r", encoding="utf-8") as reader:
            for idx, line in enumerate(reader):
                token = line.strip()
                if token:
                    vocab[token] = idx
        return vocab

    def _clean_text(self, text: str):
        text = str(text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'<[^>]*>', '', text)
        
        target = r"[^가-힣0-9a-zA-Z.,!?'\" ]"
        text = re.sub(target, repl=" ", string=text)

        text = re.sub(r'([.,!?"\'])(\1{1,})', r'\1', text)
        text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r"\s+", repl=" ", string=text)
        
        return text.strip()

    def _is_control(self, char):
        return unicodedata.category(char) in ["Cc", "Cf"]

    def _is_punctuation(self, char):
        cp = ord(char)
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3040 and cp <= 0x309F) or
            (cp >= 0x30A0 and cp <= 0x30FF) or
            (cp >= 0xAC00 and cp <= 0xD7A3)):
            return False

        if ((char >= '!' and char <= '/') or
            (char >= ':' and char <= '@') or
            (char >= '[' and char <= '`') or
            (char >= '{' and char <= '~') or
            (char == '、') or (char == '。') or (char == '《') or (char == '》') or
            (char == '「') or (char == '」') or (char == '『') or (char == '』') or
            (char == '【') or (char == '】') or (char == '〔') or (char == '〕') or
            (char == '―') or (char == '…') or (char == '–') or (char == '—')):
            return True
        
        return unicodedata.category(char).startswith("P")

    def _is_whitespace(self, char):
        return char.isspace() or ord(char) == 0x3000

    def _wordpiece_split(self, token):
        output_tokens = []
        if self.unk_token_id == -1:
            output_tokens.append(self.unk_token)
            return output_tokens

        if not token:
            return output_tokens

        start = 0
        while start < len(token):
            end = len(token)
            current_substr = None
            found = False
            while start < end:
                sub = token[start:end]
                if start > 0:
                    sub = "##" + sub
                
                if sub in self.vocab:
                    current_substr = sub
                    found = True
                    break
                end -= 1

            if not found:
                output_tokens.append(self.unk_token)
                break
            
            output_tokens.append(current_substr)
            start = end
        return output_tokens

    def get_vocab_size(self):
        return len(self.vocab)
    
    def tokenize(self, text):
        if self.clean_text:
            text = self._clean_text(text)
        if self.do_lower_case:
            text = text.lower()

        basic_tokens = text.split()

        tokens = []
        for basic_token in basic_tokens:
            if self.strip_accents:
                basic_token = unicodedata.normalize("NFD", basic_token)
                basic_token = "".join([c for c in basic_token if unicodedata.category(c) != "Mn"])

            tokens.extend(self._wordpiece_split(basic_token))
        return tokens

    def encode(self, text, max_length, add_special_tokens=True, padding=True, truncation=True):
        tokens = self.tokenize(text)
        
        final_tokens = []
        if add_special_tokens:
            final_tokens.append(self.cls_token)
        final_tokens.extend(tokens)
        if add_special_tokens:
            final_tokens.append(self.sep_token)

        if truncation and len(final_tokens) > max_length:
            if add_special_tokens:
                final_tokens = final_tokens[:max_length - 1] + [self.sep_token]
            else:
                final_tokens = final_tokens[:max_length]

        input_ids = [self.vocab.get(token, self.unk_token_id) for token in final_tokens]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        if padding:
            pad_count = max_length - len(input_ids)
            if pad_count > 0:
                input_ids.extend([self.pad_token_id] * pad_count)
                attention_mask.extend([0] * pad_count)
                token_type_ids.extend([0] * pad_count)

        if len(input_ids) != max_length:
            print(f"경고: 최종 입력 길이({len(input_ids)})가 예상 최대 길이({max_length})와 일치하지 않습니다. 잘림/패딩 로직을 확인하세요.")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

    def decode(self, input_ids):
        tokens = []
        for _id in input_ids:
            if _id in self.ids_to_tokens:
                token = self.ids_to_tokens[_id]
                if token == self.cls_token:
                    tokens.append("[CLS]")
                elif token == self.sep_token:
                    tokens.append("[SEP]")
                elif token == self.pad_token:
                    tokens.append("[PAD]")
                elif token == self.unk_token:
                    tokens.append("[UNK]")
                elif token == self.mask_token:
                    tokens.append("[MASK]")
                elif token.startswith("##"):
                    tokens.append(token[2:])
                else:
                    tokens.append(" " + token)
            else:
                tokens.append("[UNK]")
        return "".join(tokens).strip()

class CustomDataCollatorForMLM:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]

    def __call__(self, examples):

        filtered_examples = [ex for ex in examples if ex['input_ids'].numel() > 0]
        
        if not filtered_examples:
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "token_type_ids": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long),
            }

        max_length = max([len(ex['input_ids']) for ex in filtered_examples])

        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []

        for ex in filtered_examples:
            input_ids = ex['input_ids']
            attention_mask = ex['attention_mask']
            token_type_ids = ex['token_type_ids']

            padding_length = max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            token_type_ids = torch.cat([token_type_ids, torch.zeros(padding_length, dtype=torch.long)])

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
        
        input_ids = torch.stack(batch_input_ids)
        attention_mask = torch.stack(batch_attention_mask)
        token_type_ids = torch.stack(batch_token_type_ids)

        input_ids, labels = self.mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

    def mask_tokens(self, inputs: torch.Tensor):
        labels = inputs.clone()
        
        probability_matrix = torch.full_like(labels, self.mlm_probability, dtype=torch.float)
        
        special_tokens_tensor = torch.tensor(self.special_tokens_mask, device=inputs.device)
        is_special_token_mask = torch.isin(inputs, special_tokens_tensor)
        
        probability_matrix.masked_fill_(is_special_token_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full_like(labels, 0.8, dtype=torch.float)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full_like(labels, 0.5, dtype=torch.float)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.get_vocab_size(), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels