from transformers import pipeline

class TextSummarizer:
    #constructor - special method that runs automatically when you create an object
    def __init__(self,
                 model_name: str = "facebook/bart-large-cnn",
                 max_length: int = 60, # :int is a type hint -> not required but helpful for readability
                 min_length: int = 20,
                 do_sample: bool = False):
        """
        Initialize the summarization pipeline.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample

        # Load the Hugging Face summarization pipeline
        self._pipeline = pipeline(
            "summarization",
            model=self.model_name
        )

    def summarize(self, text: str) -> str:
        """
        Summarize the given text and return the summary string.
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        result = self._pipeline(
            text,
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=self.do_sample,
            truncation=True
        )

        return result[0]["summary_text"]
    
def compute_length_reduction(original: str, summary: str):

    orig_len = len(original.split()) #word count of original text
    sum_len = len(summary.split()) if summary else 0 #word count of summary

    if orig_len == 0:
        return 0, sum_len, 0.0

    reduction = 1 - (sum_len / orig_len) #fraction reduced (e.g., 0.6 means 60% shorter)
    return orig_len, sum_len, reduction
            
                     


            