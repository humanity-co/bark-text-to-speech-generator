import re

class TextProcessor:
    """
    Handles text normalization and tokenization for the Bark-style model.
    Includes support for special tokens: [laugh], [sigh], [breath], [pause].
    """
    def __init__(self):
        # Basic mapping for special tokens
        self.special_tokens = {
            "[laugh]": 1,
            "[sigh]": 2,
            "[breath]": 3,
            "[pause]": 4,
            # Add more as needed
        }
        # Reverse mapping
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Start regular tokens from offset
        self.token_offset = 100
        
    def normalize_text(self, text: str, lang: str = "en-us"):
        """
        Clean and normalize text before tokenization.
        Supports multilingual normalization.
        """
        text = text.lower().strip()
        # Keep special tokens but isolate them
        for token in self.special_tokens:
            text = text.replace(token, f" {token} ")
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    def phonemize(self, text: str, lang: str = "en-us"):
        """
        Convert text to phonemes for richer expressiveness.
        Supports: 'en-us', 'hi' (Hindi), 'mr' (Marathi)
        """
        # In a real project, we would use:
        # from phonemizer import phonemize
        # return phonemize(text, language=lang, backend='espeak')
        
        # Placeholder for implementation-grade phonetic estimation
        # This structure allows the Semantic model to learn phoneme-to-audio mappings
        return text # For now, return normalized text as "character phonemes"

    def tokenize(self, text: str, lang: str = "mr", use_phonemes: bool = False):
        """
        Tokenize text, with specific support for Devanagari (Marathi).
        """
        # For Marathi fine-tuning, we often use character-level tokens for simplicity
        # or a byte-level encoding if using pre-trained weights.
        text = self.normalize_text(text, lang=lang)
        tokens = []
        
        parts = text.split()
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                for char in part:
                    # Devanagari characters (Marathi) are in the range U+0900 to U+097F
                    # We ensure these ordinals are captured.
                    tokens.append(ord(char)) 
                tokens.append(ord(' '))
                
        return tokens

    def detokenize(self, tokens: list):
        """
        Convert tokens back to text.
        """
        text = ""
        for t in tokens:
            if t in self.inv_special_tokens:
                text += self.inv_special_tokens[t] + " "
            else:
                text += chr(t - self.token_offset)
        return text.strip()

if __name__ == "__main__":
    processor = TextProcessor()
    text = "Hello world! [laugh] This is a test [pause]."
    tokens = processor.tokenize(text)
    print(f"Tokens: {tokens}")
    print(f"Detokenized: {processor.detokenize(tokens)}")
