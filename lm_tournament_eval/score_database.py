import sqlite3
import logging

from lm_tournament_eval.api.match import (
    Match, 
    InstanceRecord, 
    InstanceUpdate,
)

from typing import List

_MODEL_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS MODEL (
    model_name TEXT NOT NULL,
    quantization_level TEXT NOT NULL,
    model_args TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (model_name, quantization_level, model_args)
);
"""

_TASK_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS TASK (
    task_name TEXT NOT NULL UNIQUE,
    output_type TEXT CHECK( output_type IN ('loglikelihood', 'loglikelihood_rolling', 'multiple_choice', 'generate_until') ) NOT NULL,
    num_instances INTEGER NOT NULL,
    PRIMARY KEY (task_name)
);
"""

_INSTANCE_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS INSTANCE (
    task_name INTEGER NOT NULL,
    doc_id INTEGER NOT NULL,
    doc_hash TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    target_hash TEXT NOT NULL,

    PRIMARY KEY (task_name, doc_id),
    FOREIGN KEY (task_name) REFERENCES TASK(task_name),
    UNIQUE (task_name, doc_id)
);
"""

_TOURNAMENT_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS TOURNAMENT (
    tournament_name TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    numpy_random_seed INTEGER NOT NULL,
    torch_random_seed INTEGER NOT NULL,
    limit_value INTEGER,
    filter_value TEXT NOT NULL DEFAULT '[none]',
    num_rounds INTEGER NOT NULL DEFAULT 1,
    batch_size INTEGER NOT NULL DEFAULT 1,
    gen_kwargs TEXT,
    match_size INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (tournament_name)
);
"""

_MATCH_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS MATCH (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_name TEXT NOT NULL,
    model0 TEXT NOT NULL,
    model0_quantization TEXT NOT NULL,
    model0_args TEXT,
    model1 TEXT NOT NULL,
    model1_quantization TEST NOT NULL,
    model1_args TEXT,
    task TEXT NOT NULL,
    match_size INT NOT NULL,
    schedule TEXT NOT NULL,
    FOREIGN KEY (tournament_name) REFERENCES TOURNAMENT(tournament_name),
    FOREIGN KEY (model0, model0_quantization, model0_args) REFERENCES MODEL(model_name, quantization_level, model_args),
    FOREIGN KEY (model1, model1_quantization, model1_args) REFERENCES MODEL(model_name, quantization_level, model_args)
);
"""

_MATCH_SCHEDULE_DEF = """
CREATE TABLE IF NOT EXISTS MATCH_SCHEDULE (
    match_id INTEGER NOT NULL,
    schedule_index INTEGER NOT NULL,
    PRIMARY KEY (match_id, schedule_index)
    FOREIGN KEY (match_id) REFERENCES MATCH(match_id)
);
"""

_INSTANCE_UPDATE_TABLE_DEF = """
CREATE TABLE IF NOT EXISTS INSTANCE_UPDATE (
    match_id INTEGER NOT NULL,

    task_name TEXT NOT NULL,
    doc_id INTEGER NOT NULL,

    model0_name TEXT NOT NULL,
    model0_quant TEXT NOT NULL,
    model0_args TEXT NOT NULL,

    model1_name TEXT NOT NULL,
    model1_quant TEXT NOT NULL,
    model1_args TEXT NOT NULL,

    model0_elo REAL NOT NULL,
    model1_elo REAL NOT NULL,

    winner TEXT CHECK(winner IN ('model0', 'model1', 'draw')),

    FOREIGN KEY (match_id) REFERENCES MATCH(match_id),
    FOREIGN KEY (task_name, doc_id) REFERENCES INSTANCE(task_name, doc_id),
    FOREIGN KEY (model0_name, model0_quant, model0_args) REFERENCES MODEL(model_name, quantization_level, model_args),
    FOREIGN KEY (model1_name, model1_quant, model1_args) REFERENCES MODEL(model_name, quantization_level, model_args)
);
"""

_INDEX_DEFS = [
"CREATE INDEX IF NOT EXISTS idx_task_dataset_name ON TASK(task_name);",
"CREATE INDEX IF NOT EXISTS idx_tournament_name ON TOURNAMENT(tournament_name);",
]

_INSERT_TOURNAMENT = """
INSERT INTO TOURNAMENT (
    tournament_name, random_seed, numpy_random_seed, torch_random_seed, limit_value,
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
    cursor.execute(_MATCH_SCHEDULE_DEF)
    cursor.execute(_INSTANCE_UPDATE_TABLE_DEF)

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
            logging.info(f"cmd_filter: {t.config.cmd_filter}")
            self.cursor.execute(_INSERT_TOURNAMENT,
                                (t.config.name, t.config.random_seed, t.config.numpy_random_seed, 
                                t.config.torch_random_seed, t.config.limit, str(t.config.cmd_filter), t.config.rounds, 
                                t.config.batch_size, t.config.gen_kwargs, t.config.match_size)
                            )
            self.database.commit()
        except sqlite3.Error as e:
            logging.error(f"Error recording tournament: {e}")
            self.database.rollback()

    def get_tournament(self, t):
        try:
            self.cursor.execute("""
            
            """, ())
        except sqlite3.Error as e:
            logging.error(f"Error getting tournament: {e}")

    def check_model_exists(self, model, quantization, model_args):
        try:
            if model_args is None:
                model_args = "None"
            self.cursor.execute("""
            SELECT 1 FROM MODEL 
            WHERE model_name = ? AND quantization_level = ? AND model_args = ?
            """, (model, quantization, model_args))
            
            result = self.cursor.fetchone()
            return result is not None
        
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return False
        
    def insert_model(self, model, quantization, model_args, score=1200):
        try:
            if model_args is None:
                model_args = "None"
            self.cursor.execute("""
            INSERT INTO MODEL (model_name, quantization_level, model_args, score)
            VALUES (?, ?, ?, ?)
            """, (model, quantization, model_args, score))
            
            self.database.commit()
            print(f"Model {model} with quantization level {quantization} inserted successfully.")
            return True
        
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error: {e}")
            print(f"Model {model} with quantization level {quantization} already exists.")
            self.database.rollback()
            return False
        
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            self.database.rollback()
            return False

    def get_model_score(self, name, quantization_level, model_args):
        try:
            self.cursor.execute("""
            SELECT score FROM MODEL 
            WHERE model_name = ? AND quantization_level = ? AND model_args = ?
            """, (name, quantization_level, model_args))
            
            result = self.cursor.fetchone()
            return result[0] if result else None
        
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None

    def set_model_score(self, name, quantization_level, model_args, new_score):
        try:
            self.cursor.execute("""
            UPDATE MODEL 
            SET score = ? 
            WHERE model_name = ? AND quantization_level = ? AND model_args = ?
            """, (new_score, name, quantization_level, model_args))
            
            if self.cursor.rowcount == 0:
                print(f"Model {name} with quantization level {quantization_level} not found.")
                return False
            
            self.database.commit()
            print(f"Score updated for model {name} with quantization level {quantization_level}.")
            return True
        
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            self.database.rollback()
            return False

    def task_exists(self, task_name):
        """
        Check if a task exists in the database.
        
        :param conn: SQLite database connection
        :param task_name: Name of the task to check
        :return: True if the task exists, False otherwise
        """        
        self.cursor.execute("SELECT 1 FROM TASK WHERE task_name = ?", (task_name,))
        return self.cursor.fetchone() is not None
    
    def insert_task(self, task_name, output_type, num_instances):
        try:
            self.cursor.execute("""
                INSERT INTO TASK (task_name, output_type, num_instances)
                VALUES (?, ?, ?)
            """, (task_name, output_type, num_instances))
            self.database.commit()
            return True
        except sqlite3.IntegrityError:
            self.database.rollback()
            return False

    def match_exists(self, m : Match):
        tournament_name = m.tournament_name
        model0_key = m.model0_key
        model1_key = m.model1_key

        query = """
        SELECT 1 FROM MATCH 
        WHERE tournament_name = ? AND model0 = ? AND model0_quantization = ? AND model0_args = ? AND model1 = ? AND model1_quantization = ? AND model1_args = ?
        """
        params = [tournament_name, *model0_key, *model1_key]

        self.cursor.execute(query, params)
        return self.cursor.fetchone() is not None

    def insert_match_schedule(self, match_id, schedule_index):
        try:
            self.cursor.execute("""
                INSERT INTO MATCH_SCHEDULE (match_id, schedule_index) VALUES (?, ?)
            """, (match_id, schedule_index))
            self.database.commit()

            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            self.database.rollback()
            return None
        
    def instance_exists(self, ir: InstanceRecord) -> bool:
        
        query = """
        SELECT 1 FROM INSTANCE
        WHERE task_name = ? AND doc_id = ?
        """
        params = [ir.task_name, ir.doc_id]
        self.cursor.execute(query, params)
        return self.cursor.fetchone() is not None

    def insert_instance(self, ir : InstanceRecord):
        try:
            self.cursor.execute("""
            INSERT INTO INSTANCE (task_name, doc_id, doc_hash, prompt_hash, target_hash)
            VALUES (?, ?, ?, ?, ?)
            """, (ir.task_name, ir.doc_id, ir.doc_hash, ir.prompt_hash, ir.target_hash)
            )
            self.database.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            self.database.rollback()
            return None

    def insert_instance_update(self, iu : InstanceUpdate):
        try:
            self.cursor.execute("""
            INSERT INTO INSTANCE_UPDATE (match_id, task_name, doc_id, model0_name, model0_quant, model0_args, model1_name, model1_quant, model1_args, model0_elo, model1_elo, winner) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (iu.match_id, iu.task_name, iu.doc_id, iu.model0_name, iu.model0_quantization, iu.model0_args, iu.model1_name, iu.model1_quantization, iu.model1_args, iu.model0_elo, iu.model1_elo, iu.winner))
            
            self.database.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError as e:
            print(f"instance_update insert error: {e}")
            self.database.rollback()
            return None

    def update_instance_records(self, match : Match, samples0, samples1):
        for schedule_idx, sample_0, sample_1 in zip(match.schedule, samples0, samples1):
            assert(sample_0['doc_id'] == sample_1['doc_id'])

            instance_record = InstanceRecord(
                task_name=match.task,
                doc_id=schedule_idx,
                doc_hash=sample_0['doc_hash'],
                prompt_hash=sample_0['prompt_hash'],
                target_hash=sample_0['target_hash']
            )

            if not self.instance_exists(instance_record):
                self.insert_instance(instance_record)

    def record_instance_updates(self, 
                                match : Match, 
                                match_id : int, 
                                samples0, 
                                samples1, 
                                elo_0 : float, 
                                elo_1 : float, 
                                winners : List[str]):

        for idx, (sample_0, sample_1) in enumerate(zip(samples0, samples1)):
            assert(sample_0['doc_id'] == sample_1['doc_id'])

            instance_update = InstanceUpdate(
                match_id=match_id,
                task_name=match.task,
                doc_id = match.schedule[idx],
                model0_name = match.model0_key[0],
                model0_quantization=match.model0_key[1],
                model0_args=match.model0_key[2],
                model0_elo=elo_0,
                model1_name = match.model1_key[0],
                model1_quantization=match.model1_key[1],
                model1_args=match.model1_key[2],
                model1_elo=elo_1,
                winner=winners[idx]
            )

            self.insert_instance_update(instance_update)

    def insert_match(self, m : Match):
        tournament_name = m.tournament_name
        model0_key = m.model0_key
        model1_key = m.model1_key
        task = m.task
        match_size = m.match_size
        schedule = m.schedule
        try:
            self.cursor.execute("""
                INSERT INTO MATCH (tournament_name, model0, model0_quantization, model0_args, model1, model1_quantization, model1_args, task, match_size, schedule)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (tournament_name, *model0_key, *model1_key, task, match_size, str(schedule)))
            self.database.commit()
            id = self.cursor.lastrowid

            for index in m.schedule:
                self.insert_match_schedule(id, index)

            return id
        except sqlite3.IntegrityError:
            self.database.rollback()
            return None



