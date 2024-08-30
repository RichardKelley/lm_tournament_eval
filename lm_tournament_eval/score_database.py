import sqlite3
import logging

_MODEL_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS MODEL (
    name TEXT NOT NULL,
    quantization_level TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (name, quantization_level)
);
"""

_TASK_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS TASK (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL UNIQUE,
    type TEXT CHECK( type IN ('loglikelihood', 'loglikelihood_rolling', 'multiple_choice') ) NOT NULL,
    num_training_instances INTEGER NOT NULL,
    num_validation_instances INTEGER NOT NULL,
    num_test_instances INTEGER NOT NULL
);
"""

_INSTANCE_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS INSTANCE (
    instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    index_within_task INTEGER NOT NULL,
    text TEXT NOT NULL,
    target TEXT NOT NULL,
    split TEXT CHECK( split IN ('train', 'validation', 'test') ) NOT NULL,
    FOREIGN KEY (task_id) REFERENCES TASK(task_id),
    UNIQUE (task_id, index_within_task)
);
"""

_TOURNAMENT_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS TOURNAMENT (
    tournament_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    numpy_random_seed INTEGER NOT NULL,
    torch_random_seed INTEGER NOT NULL,
    limit_value INTEGER,
    filter_value TEXT NOT NULL DEFAULT 'none',
    num_rounds INTEGER NOT NULL DEFAULT 1,
    batch_size INTEGER NOT NULL DEFAULT 1,
    gen_kwargs TEXT,
    match_size INTEGER NOT NULL DEFAULT 1
);
"""

_MATCH_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS MATCH (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL,
    model0 TEXT NOT NULL,
    model0_args TEXT,
    model1 TEXT NOT NULL,
    model1_args TEXT,
    tasks TEXT,
    FOREIGN KEY (tournament_id) REFERENCES TOURNAMENT(tournament_id),
    FOREIGN KEY (model0) REFERENCES MODEL(name),
    FOREIGN KEY (model1) REFERENCES MODEL(name)
);
"""

_MATCH_TASK_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS MATCH_TASK (
    match_id INTEGER NOT NULL,
    task_id INTEGER NOT NULL,
    PRIMARY KEY (match_id, task_id),
    FOREIGN KEY (match_id) REFERENCES MATCH(match_id),
    FOREIGN KEY (task_id) REFERENCES TASK(task_id)
);
"""

_MATCH_RESULT_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS MATCH_RESULT (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    model0_score REAL,
    model1_score REAL,
    winner TEXT CHECK( winner IN ('model0', 'model1', 'tie') ),
    FOREIGN KEY (match_id) REFERENCES MATCH(match_id),
    FOREIGN KEY (instance_id) REFERENCES INSTANCE(instance_id)
);
"""

_INDEX_DEFS = [
"CREATE INDEX IF NOT EXISTS idx_task_dataset_name ON TASK(dataset_name);",
"CREATE INDEX IF NOT EXISTS idx_instance_task ON INSTANCE(task_id);",
"CREATE INDEX IF NOT EXISTS idx_instance_split ON INSTANCE(split);",
"CREATE INDEX IF NOT EXISTS idx_tournament_name ON TOURNAMENT(name);",
"CREATE INDEX IF NOT EXISTS idx_match_tournament ON MATCH(tournament_id);",
"CREATE INDEX IF NOT EXISTS idx_match_models ON MATCH(model0, model1);",
"CREATE INDEX IF NOT EXISTS idx_match_result_match ON MATCH_RESULT(match_id);",
"CREATE INDEX IF NOT EXISTS idx_match_result_instance ON MATCH_RESULT(instance_id);"
]

_INSERT_TOURNAMENT = """
INSERT INTO TOURNAMENT (
    name, random_seed, numpy_random_seed, torch_random_seed, limit_value,
    filter_value, num_rounds, batch_size, gen_kwargs, match_size
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

def _table_exists(cursor, table_name):
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

def _initialize_database(cursor):

    # enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")

    # set up the database tables if necessary.
    cursor.execute(_MODEL_TABLE_DEF)
    cursor.execute(_TASK_TABLE_DEF)
    cursor.execute(_INSTANCE_TABLE_DEF)
    cursor.execute(_TOURNAMENT_TABLE_DEF)
    cursor.execute(_MATCH_TABLE_DEF)
    cursor.execute(_MATCH_TASK_TABLE_DEF)
    cursor.execute(_MATCH_RESULT_TABLE_DEF)

    # create indexes
    cursor.execute("BEGIN TRANSACTION")
    for command in _INDEX_DEFS:
        cursor.execute(command)

class ScoreDatabase:
    def __init__(self, dbpath):
        self.dbpath = dbpath
        self.database = sqlite3.connect(self.dbpath)
        self.cursor = self.database.cursor()

        _initialize_database(self.cursor)

    def record_tournament(self, t):
        logging.info("Recording tournament.")
        try:
            self.cursor.execute(_INSERT_TOURNAMENT,
                                (t.config.name, t.config.random_seed, t.config.numpy_random_seed, 
                                t.config.torch_random_seed, t.config.limit, t.config.cmd_filter, t.config.rounds, 
                                t.config.batch_size, t.config.gen_kwargs, t.config.match_size)
                            )
            self.database.commit()
        except sqlite3.Error as e:
            logging.error(f"Error recording tournament: {e}")
            self.database.rollback()

    def check_model_exists(self, model, quantization):
        try:
            self.cursor.execute("""
            SELECT 1 FROM MODEL 
            WHERE name = ? AND quantization_level = ?
            """, (model, quantization))
            
            result = self.cursor.fetchone()
            return result is not None
        
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return False
        
    def insert_model(self, model, quantization, score=1200):
        try:
            self.cursor.execute("""
            INSERT INTO MODEL (name, quantization_level, score)
            VALUES (?, ?, ?)
            """, (model, quantization, score))
            
            self.database.commit()
            print(f"Model {model} with quantization level {quantization} inserted successfully.")
            return True
        
        except sqlite3.IntegrityError:
            print(f"Model {model} with quantization level {quantization} already exists.")
            self.database.rollback()
            return False
        
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            self.database.rollback()
            return False

    def get_model_score(self, model_name, quantization):
        pass

    def set_model_score(self, model_name, quantization, model_score):
        pass

    def get_task_instance_scores(self, task_name, instance_idx):
        pass

    def set_task_instance_score(self, task_name, instance_idx, model_name, score):
        pass

    def get_task_average_score(self, task_name):
        # this should be computed from (task, instance, model, score) tuples
        pass




