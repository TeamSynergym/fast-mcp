import csv
from app.services.exercise_vector_db import ExerciseVectorDB

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_exercise_data(csv_path):
  documents = []
  with open(csv_path, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    print('CSV Headers:', reader.fieldnames)  # Debug: print headers
    for row in reader:
      documents.append({
        'name': row.get('name'),
        'description': row.get('description'),
        'difficulty': row.get('difficulty'),
        'body_part': row.get('body_part')
      })
  return documents


if __name__ == "__main__":
  data = load_exercise_data('data/naver_exercise_cleaned.csv')
  vector_db = ExerciseVectorDB()
  vector_db.build_index(data)
  print("Exercise vector database built successfully.")