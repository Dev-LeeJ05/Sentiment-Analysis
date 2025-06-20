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

    def _load_vocab(self, vocab_file_path):
        vocab = {}
        with open(vocab_file_path, "r", encoding="utf-8") as reader:
            for idx, line in enumerate(reader):
                token = line.strip()
                if token:
                    vocab[token] = idx
        return vocab

    def _clean_text(self, text):
        output = []
        for char in text:
            if self._is_control(char) or self._is_punctuation(char):
                output.append(" ")
            elif self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return re.sub(r'\s+', ' ', "".join(output)).strip()

    def _is_control(self, char):
        return unicodedata.category(char) in ["Cc", "Cf"]

    def _is_punctuation(self, char):
        return unicodedata.category(char).startswith("P") or (
            (char >= '!' and char <= '/') or
            (char >= ':' and char <= '@') or
            (char >= '[' and char <= '`') or
            (char >= '{' and char <= '~') or
            (char == '、') or (char == '。') or (char == '《') or (char == '》') or
            (char == '「') or (char == '」') or (char == '『') or (char == '』') or
            (char == '【') or (char == '】') or (char == '〔') or (char == '〕') or
            (char == '―') or (char == '…') or (char == '–') or (char == '—')
        )

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

    def tokenize(self, text):
        if self.clean_text:
            text = self._clean_text(text)
        if self.do_lower_case:
            text = text.lower()

        basic_tokens = text.split()

        tokens = []
        for basic_token in basic_tokens:
            tokens.extend(self._wordpiece_split(basic_token))
        return tokens

    def encode(self, text, max_length, add_special_tokens=True, padding=True, truncation=True):
        tokens = self.tokenize(text)
        input_ids = []
        attention_mask = []
        token_type_ids = []

        if add_special_tokens:
            input_ids.append(self.cls_token_id)
            attention_mask.append(1)
            token_type_ids.append(0)

        for token in tokens:
            input_ids.append(self.vocab.get(token, self.unk_token_id))
            attention_mask.append(1)
            token_type_ids.append(0)

        if add_special_tokens:
            input_ids.append(self.sep_token_id)
            attention_mask.append(1)
            token_type_ids.append(0)
        
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

        if padding and len(input_ids) < max_length:
            pad_count = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_count)
            attention_mask.extend([0] * pad_count)
            token_type_ids.extend([0] * pad_count)

        if len(input_ids) != max_length:
            print(f"경고: 입력 길이({len(input_ids)})가 최대 길이({max_length})와 일치하지 않습니다. 잘림/패딩 로직에 문제가 있을 수 있습니다.")

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
                if token in [self.cls_token, self.sep_token, self.pad_token, self.unk_token, self.mask_token]:
                    continue
                if token.startswith("##"):
                    tokens.append(token[2:])
                else:
                    tokens.append(" " + token)
            else:
                tokens.append(self.unk_token)
        return "".join(tokens).strip()
        