"""Определение класса Predictor"""
import os
import time
import multiprocessing as mp
import pickle
import random  # Для тестирования
import numpy as np
import pandas as pd
from logger import init_stream_logger


class Predictor:
    """Возвращает предсказания модели"""

    def __init__(
        self, scanner=None, preproccessor=None, models_dir="models", period=60, logger=None
    ):
        self.scanner = scanner
        self.preproccessor = preproccessor
        self.models_dir = models_dir
        self.period = period
        self.logger = logger

    def _predict(self, model, log_file, min_bound):
        """Выполняет предсказание"""

        self.logger.info("Prediction process starts")

        while True:
            (
                sessions, ip, data
            ) = self.preproccessor.proccess_data(log_file)

            if len(data):
                self.logger.debug(f"Prepared records for predictions: {len(data)}")
                try:
                    predictions = model.predict_proba(
                        np.array(data))[:, 1]
                    self.logger.debug("Prediction made")
                except Exception as e:
                    self.logger.error(f"Prediction error: {e}")

                # df_out = pd.DataFrame(columns=["timestamp", "ip_addr", "prob", "session"])
                # df_out["timestamp"] = timestamps
                # df_out["ip_addr"] = ip_addrs
                # df_out["session"] = sessions
                # df_out["user_agent"] = user_agents
                # df_out["prob"] = np.round(predictions, 2)

                df_out = pd.DataFrame(columns=["prob", "ip", "session"])
                df_out["session"] = sessions
                df_out["ip"] = ip
                df_out["prob"] = np.round(predictions, 2)

                # Для тестирования
                # predictions = [random.random()
                #                for _ in range(len(data))]
                # predictions_filtered = [
                #     predict for predict in predictions if predict > min_bound
                # ]

                df_out_filtered = df_out[df_out["prob"] > min_bound]
                self.logger.debug(f"Predicted bots: {len(df_out_filtered)}")

                if len(df_out_filtered) > 0:
                    out_file = os.path.join(os.path.dirname(log_file), "out.csv.log")
                    df_out_filtered.to_csv(
                        out_file, mode="a", index=False, header=False
                    )

            time.sleep(self.period)

    def _worker(self, args):
        """Вспомогательная функция, которая разбирает аргументы"""

        model_name, log_file, min_bound = args
        logger = init_stream_logger()
        logger.info("Proccess starts")
        model_path = os.path.join(self.models_dir, model_name + ".pkl")

        if os.path.exists(model_path):
            logger.info(f"Model file {model_path} found")
        else:
            logger.error(f"Model file {model_path} not found")

        try:
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            logger.info(f"Model from file {model_path} was loaded")
        except pickle.PickleError as e:
            logger.error(f"Model from file {model_path} was not loaded")

        return self._predict(model, log_file, min_bound)

    def start(self):
        """Запускает процесс предсказания"""

        models_names = self.scanner.get_models()
        log_files = self.scanner.get_log_files()
        min_bounds = self.scanner.get_min_bounds()

        num_procs = len(log_files)  # число процессов = число файлов с логами
        self.logger.info(f"Number of processes: {num_procs}")

        with mp.Pool(num_procs) as pool:
            pool.map(self._worker, zip(models_names, log_files, min_bounds))
