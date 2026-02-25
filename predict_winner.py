"""CLI entry point – uses prediction module."""
from prediction import run_predictions


def main():
    result = run_predictions()
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    lr = result["last_race"]
    nr = result["next_race"]
    print(f"Predicted winner for round {lr['round']} is {lr['predicted_winner']} "
          f"(actual: {lr['actual_winner']})")
    print(f"Predicted winner for next round {nr['round']} ({nr['year']}) is "
          f"{nr['predicted_winner']} with probability {nr['top_probability']:.3f}")
    print(f"Overall accuracy {result['accuracy']:.3f}")


if __name__ == "__main__":
    main()
