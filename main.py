from app.services.exercise_vector_db import ExerciseVectorDB


def main():
    print("Hello from fast-mcp!")


if __name__ == "__main__":
    vector_db = ExerciseVectorDB()

    query = "목 완화에 좋은 중급 운동"

    result = vector_db.search(query)

    print(f"추천 운동 : :", result)