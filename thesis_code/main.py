from thesis_code.utils.helpers import load_data

def main():
    print("🔹 Running Master's Thesis Analysis 🔹")

    # Load dataset
    data = load_data("data/market_data.csv")

    if data is not None:
        # Compute daily returns
        data["returns"] = data["price"].pct_change()
        data = data.dropna()

        # Run regime-switching model
        

        #if results:
        #    print(results.summary())

if __name__ == "__main__":
    main()
