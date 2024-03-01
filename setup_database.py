from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Float, DateTime


def create_database(database_path='sqlite:///sentiment_analysis_db.sqlite'):
    engine = create_engine(database_path)
    metadata = MetaData()

    models = Table('models', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('model_name', String),
                   Column('model_path', String),
                   Column('vectorizer_path', String),
                   Column('created_at', DateTime),
                   Column('version', String))

    metrics = Table('metrics', metadata,
                    Column('id', Integer, primary_key=True),
                    Column('accuracy', Float),
                    Column('precision', Float),
                    Column('recall', Float),
                    Column('f1_score', Float),
                    Column('created_at', DateTime))
    
    statistics = Table('statistics', metadata,
                       Column('id', Integer, primary_key=True),
                       Column('positive_test', Integer),
                       Column('negative_test', Integer),
                       Column('positive_pred', Integer),
                       Column('negative_pred', Integer),
                       Column('created_at', DateTime))

    failure_stats_table = Table('failure_statistics', metadata,
                                Column('id', Integer, primary_key=True),
                                Column('rating', Integer),
                                Column('incorrect_predictions', Integer),
                                Column('created_at', DateTime))

    # New table for storing reviews and their ratings
    reviews = Table('reviews', metadata,
                    Column('id', Integer, primary_key=True),
                    Column('review_text', String),
                    Column('rating', Integer),  # 1 for positive, 0 for negative
                    Column('created_at', DateTime))

    metadata.create_all(engine)


if __name__ == '__main__':
    create_database()
