import argparse, nltk, evaluate
from summarizer import TextSummarizer


# Ensures sentence tokenizer exists that is used for ROUGE-Lsum 
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


SAMPLE_DATA: List[Dict[str, str]] = [
    {
        "id": "sample_1",
        "text": (
            "The company announced a new product update on Monday, introducing features aimed at improving "
            "performance and reliability. Executives said the update was shaped by customer feedback and "
            "would roll out gradually over the coming weeks. Analysts expect the changes to strengthen "
            "the product’s competitiveness, though they cautioned that adoption may vary across regions."
        ),
        "reference": (
            "The company released a product update with new performance and reliability features, rolling "
            "out over the next few weeks after customer feedback."
        ),
    },
    {
        "id": "sample_2",
        "text": (
            "A research team published findings showing that improved data cleaning and feature engineering "
            "significantly boosted model stability. The authors evaluated multiple configurations and noted "
            "that even small preprocessing changes could affect accuracy. They recommended standardized "
            "evaluation protocols to improve reproducibility across future experiments."
        ),
        "reference": (
            "Researchers found that better preprocessing improves model stability and recommended standardized "
            "evaluation protocols for reproducibility."
        ),
    },
    {
        "id": "sample_3",
        "text": (
            "During the meeting, stakeholders aligned on priorities for the next quarter. The team will focus "
            "on reducing latency, improving monitoring, and refining the user onboarding flow. A follow-up "
            "session is scheduled to finalize milestones and confirm ownership across engineering and product."
        ),
        "reference": (
            "Stakeholders agreed next-quarter priorities: reduce latency, improve monitoring, and refine onboarding, "
            "with a follow-up to finalize milestones and ownership."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BrieflyAI summarizer with ROUGE.")
    p.add_argument("--model", default="bart", choices=["bart", "distilbart", "t5"], help="Model choice.")
    p.add_argument("--max", type=int, default=120, help="Max summary length (tokens).")
    p.add_argument("--min", type=int, default=40, help="Min summary length (tokens).")
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (creative mode).")
    p.add_argument("--limit", type=int, default=3, help="How many samples from the built-in set to evaluate.")
    return p.parse_args()


def build_model_name(choice: str) -> str:
    choice = choice.lower()
    if choice == "bart":
        return "facebook/bart-large-cnn"
    if choice == "distilbart":
        return "sshleifer/distilbart-cnn-12-6"
    if choice == "t5":
        return "t5-small"
    return "facebook/bart-large-cnn"


def main() -> None:
    args = parse_args()

    model_name = build_model_name(args.model)
    ts = TextSummarizer(
        model_name=model_name,
        max_length=args.max,
        min_length=args.min,
        do_sample=args.do_sample,
    )

    rouge = evaluate.load("rouge")

    samples = SAMPLE_DATA[: max(1, min(args.limit, len(SAMPLE_DATA)))]
    predictions: List[str] = []
    references: List[str] = []

    print(f"\nEvaluating ROUGE for model={args.model} ({model_name}) "
          f"max={args.max} min={args.min} do_sample={args.do_sample}\n")

    for s in samples:
        pred = ts.summarize(s["text"])
        predictions.append(pred)
        references.append(s["reference"])

        print(f"--- {s['id']} ---")
        print("PRED:", pred)
        print("REF: ", s["reference"])
        print()

    # ROUGE-Lsum expects sentences separated by newlines for best results
    # We'll replace ". " with ".\n" as a lightweight formatting trick
    predictions_lsum = [p.replace(". ", ".\n") for p in predictions]
    references_lsum = [r.replace(". ", ".\n") for r in references]

    results = rouge.compute(
        predictions=predictions_lsum,
        references=references_lsum,
        use_stemmer=True,
    )

    # results includes: rouge1, rouge2, rougeL, rougeLsum
    print("ROUGE Results:")
    print(f"ROUGE-1   : {results['rouge1']:.4f}")
    print(f"ROUGE-2   : {results['rouge2']:.4f}")
    print(f"ROUGE-L   : {results['rougeL']:.4f}")
    print(f"ROUGE-Lsum: {results['rougeLsum']:.4f}")
    print()


if __name__ == "__main__":
    main()
