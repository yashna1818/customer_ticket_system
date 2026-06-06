import pandas as pd
from data_loader import load_and_detect_data
from preprocessing import optimize_categories
from model import TicketClassifier

print("Loading data...")
df = load_and_detect_data()
print("Preprocessing categories...")
df = optimize_categories(df)

print("Training models...")
clf = TicketClassifier()
metrics = clf.train(df)
print("\nTrained metrics:")
for name, m in metrics.items():
    print(name, m)
