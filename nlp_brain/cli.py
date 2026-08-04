"""Command-line entry point for the NLP Brain, standalone from any API/voice
process: `python -m nlp_brain.cli train|predict "<text>"|interactive`."""
import sys

from .classifier import SymptomClassifier
from .config import DATA_PATH, GIBBERISH_SAMPLE_QUERIES, MODEL_OUT, NO_MATCH_MESSAGE, SAMPLE_QUERIES


def _format(result) -> str:
    return result.specialist if result.specialist else NO_MATCH_MESSAGE


def train(data_path: str = DATA_PATH, model_out: str = MODEL_OUT) -> None:
    classifier, _ = SymptomClassifier.train(data_path)
    classifier.save(model_out)
    print(f"\nSaved trained pipeline to {model_out}")

    print("\n=== Sample query predictions ===")
    for query in SAMPLE_QUERIES:
        print(f"{query!r:70s} -> {_format(classifier.predict(query))}")

    print("\n=== Gibberish sample query checks (should all say 'No matches found.') ===")
    for query in GIBBERISH_SAMPLE_QUERIES:
        print(f"{query!r:70s} -> {_format(classifier.predict(query))}")


def predict_one(text: str, model_out: str = MODEL_OUT) -> None:
    classifier = SymptomClassifier.load(model_out)
    result = classifier.predict(text)
    if result.specialist:
        print(f"Predicted specialist: {result.specialist} (similarity: {result.match_ratio:.2f})")
    elif result.flagged_gibberish:
        print("Flagged as gibberish.")
    else:
        print(NO_MATCH_MESSAGE)


def interactive(model_out: str = MODEL_OUT) -> None:
    print("Loading symptom-specialist classifier...")
    classifier = SymptomClassifier.load(model_out)
    print("\n=== Try your own queries (type 'quit' to exit) ===")
    while True:
        text = input("\nEnter symptom text: ").strip()
        if text.lower() in ("quit", "exit", ""):
            break
        result = classifier.predict(text)
        if result.specialist:
            print(f"Predicted specialist: {result.specialist}")
        elif result.flagged_gibberish:
            print("Sorry, that didn't look like a symptom description.")
        else:
            print(f"Sorry, I couldn't confidently match that (closest known: {result.closest_known_example!r}).")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] == "interactive":
        interactive()
    elif argv[0] == "train":
        train()
    elif argv[0] == "predict":
        if len(argv) < 2:
            print('Usage: python -m nlp_brain.cli predict "<symptom text>"')
            sys.exit(1)
        predict_one(" ".join(argv[1:]))
    else:
        print('Usage: python -m nlp_brain.cli [train|predict "<text>"|interactive]')
        sys.exit(1)


if __name__ == "__main__":
    main()
