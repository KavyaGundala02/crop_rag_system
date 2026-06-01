# ingestion/chunking.py

import pandas as pd
from load_data import load_dataset

def row_level_chunks(df: pd.DataFrame) -> list:
    """
    Strategy 1: Each CSV row becomes one text document.
    Example output:
    "Rice grows well with N=90, P=42, K=43, 
     temperature=20°C, humidity=82%, pH=6.5, rainfall=202mm"
    """
    
    chunks = []
    
    for _, row in df.iterrows():
        text = (
            f"{row['label'].capitalize()} grows well with "
            f"N={row['N']}, "
            f"P={row['P']}, "
            f"K={row['K']}, "
            f"temperature={row['temperature']}°C, "
            f"humidity={row['humidity']}%, "
            f"pH={row['ph']}, "
            f"rainfall={row['rainfall']}mm."
        )
        
        metadata = {
            "crop"        : row["label"],
            "N"           : float(row["N"]),
            "P"           : float(row["P"]),
            "K"           : float(row["K"]),
            "temperature" : float(row["temperature"]),
            "humidity"    : float(row["humidity"]),
            "ph"          : float(row["ph"]),
            "rainfall"    : float(row["rainfall"]),
            "strategy"    : "row_level"
        }
        
        chunks.append({
            "text"     : text,
            "metadata" : metadata
        })
    
    return chunks


def crop_aggregate_chunks(df: pd.DataFrame) -> list:
    """
    Strategy 2: Group all rows by crop label.
    Summarise mean and range values per crop.
    Example output:
    "Rice: avg N=79.9 (60-100), avg P=47.6 (30-60)..."
    """
    
    chunks = []
    
    for crop, group in df.groupby("label"):
        text = (
            f"{crop.capitalize()} crop summary: "
            f"Nitrogen(N) avg={round(group['N'].mean(),1)} "
            f"range=({round(group['N'].min(),1)}-{round(group['N'].max(),1)}), "
            f"Phosphorus(P) avg={round(group['P'].mean(),1)} "
            f"range=({round(group['P'].min(),1)}-{round(group['P'].max(),1)}), "
            f"Potassium(K) avg={round(group['K'].mean(),1)} "
            f"range=({round(group['K'].min(),1)}-{round(group['K'].max(),1)}), "
            f"Temperature avg={round(group['temperature'].mean(),1)}°C "
            f"range=({round(group['temperature'].min(),1)}-{round(group['temperature'].max(),1)}), "
            f"Humidity avg={round(group['humidity'].mean(),1)}% "
            f"range=({round(group['humidity'].min(),1)}-{round(group['humidity'].max(),1)}), "
            f"pH avg={round(group['ph'].mean(),2)} "
            f"range=({round(group['ph'].min(),2)}-{round(group['ph'].max(),2)}), "
            f"Rainfall avg={round(group['rainfall'].mean(),1)}mm "
            f"range=({round(group['rainfall'].min(),1)}-{round(group['rainfall'].max(),1)})."
        )
        
        metadata = {
            "crop"            : crop,
            "avg_N"           : round(float(group["N"].mean()), 1),
            "avg_P"           : round(float(group["P"].mean()), 1),
            "avg_K"           : round(float(group["K"].mean()), 1),
            "avg_temperature" : round(float(group["temperature"].mean()), 1),
            "avg_humidity"    : round(float(group["humidity"].mean()), 1),
            "avg_ph"          : round(float(group["ph"].mean()), 2),
            "avg_rainfall"    : round(float(group["rainfall"].mean()), 1),
            "strategy"        : "crop_aggregate"
        }
        
        chunks.append({
            "text"     : text,
            "metadata" : metadata
        })
    
    return chunks


def preview_chunks(chunks: list, n: int = 3):
    """Print first n chunks to verify output."""
    
    print(f"\n Total chunks created: {len(chunks)}")
    print(f"\n Sample chunks (first {n}):")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks[:n]):
        print(f"\n Chunk {i+1}:")
        print(f" Text     : {chunk['text']}")
        print(f" Metadata : {chunk['metadata']}")
        print("-" * 60)


if __name__ == "__main__":
    
    # Load dataset
    df = load_dataset("data/Crop_recommendation.csv")
    
    # Strategy 1 - Row level
    print("\n🔹 STRATEGY 1: ROW LEVEL CHUNKS")
    row_chunks = row_level_chunks(df)
    preview_chunks(row_chunks)
    
    # Strategy 2 - Crop aggregate
    print("\n🔹 STRATEGY 2: CROP AGGREGATE CHUNKS")
    agg_chunks = crop_aggregate_chunks(df)
    preview_chunks(agg_chunks)
    
    # Summary comparison
    print("\n📊 STRATEGY COMPARISON")
    print("=" * 60)
    print(f" Row-level chunks    : {len(row_chunks)} documents")
    print(f" Aggregate chunks    : {len(agg_chunks)} documents")
    print("\n Row-level = more specific, better for exact queries")
    print(" Aggregate = more summarised, better for general queries")