import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import datetime


def plot_dates(fyler_df: pd.DataFrame, notes_df: pd.DataFrame, mrn: int):
    note_events = notes_df[notes_df["mrn"] == mrn].drop_duplicates(["event_id"])
    fyler_events = fyler_df[fyler_df["MRN"] == mrn].drop_duplicates(["HC_EVENT_ID"])
