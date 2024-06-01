import warnings

warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor

import json
from typing import Callable, List, Union
from prefixspan import PrefixSpan
import diskcache
import numpy as np
from tqdm import tqdm, trange

from .models import Model
from .models.qa_model import QAModel
from .task_store import get_pd_schema


def apply_aggregate_function(x, aggregate_func: Union[str, Callable]):
	if isinstance(aggregate_func, Callable):
		return aggregate_func(x)
	else:
		if aggregate_func == 'mean':
			return np.mean(x)
		if aggregate_func == 'std':
			return np.std(x)
		if aggregate_func == 'max':
			return np.max(x)
		if aggregate_func == 'min':
			return np.min(x)
		return x


def find_frequent_patterns(k, df, scores=None):
	if len(df) == 0:
		return []

	df = df.reset_index(drop=True)
	cols = df.columns.to_list()
	df = df.fillna('').astype('str')
	db = [[(c, v) for c, v in zip(cols, d) if v] for d in df.values.tolist()]

	ps = PrefixSpan(db)
	patterns = ps.topk(k, closed=True)
	if scores is None:
		return patterns
	else:
		aggregated_scores = []
		scores = np.asarray(scores)
		for count, pattern in patterns:
			q = ' and '.join([f"`{k}` == {repr(v)}" for k, v in pattern])
			indices = df.query(q).index.to_numpy()
			aggregated_scores.append(np.mean(scores[indices]))
		return patterns, aggregated_scores


class TaskEvaluator:
	data_field = None

	def __init__(
			self,
			task_plan_df,
			task_generator,
			embedding_func: Callable = None,
			embedding_name: str = 'st',
			embedding_batch_size: int = 10000,
			cache_path_root: str = None,
			cache_size_limit: int = 10,  # in GB
			overwrite_embedding_cache: bool = False,
			overwrite_eval_cache: bool = False,
			overwrite_task_cache: bool = False,
			seed: int = 42,
	):
		self.task_plan_df = task_plan_df.astype(get_pd_schema({k: v for k, v in task_generator.schema.items()
															   if k in task_plan_df.columns.to_list()}))
		self.task_generator = task_generator
		self.num_tasks = len(self.task_plan_df)

		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)

		self.cache_size_limit = cache_size_limit * (2 ** 30)  # in bytes
		self.embed_cache_path = None if cache_path_root is None else f"{cache_path_root}/{embedding_name}"
		self.eval_cache_path = None if cache_path_root is None else f"{cache_path_root}/eval/{seed}"
		self.task_cache_path = None if cache_path_root is None else f"{cache_path_root}/task/{seed}"
		self.overwrite_embedding_cache = overwrite_embedding_cache
		self.overwrite_eval_cache = overwrite_eval_cache
		self.overwrite_task_cache = overwrite_task_cache

		self.embedding_func = embedding_func
		self.embedding_batch_size = embedding_batch_size
		self.embeddings = None

	def _embedding(self, pool):
		"(Abstract method) embedding task"

	def check_embedding(self):
		if self.embeddings is None:
			self.embeddings = self._embedding_task()

	def set_embeddings(self, embeddings):
		self.embeddings = embeddings

	def _embedding_task(self):
		if self.embedding_func is None:
			from sentence_transformers import SentenceTransformer
			sentence_transformer = SentenceTransformer("all-mpnet-base-v2", device='cpu')
			self.embedding_func = lambda doc: sentence_transformer.encode(doc)

		embeds = []

		if self.embed_cache_path is None:
			pool = []
			for i in trange(self.num_tasks, desc="Embedding tasks"):
				plan = self._plan_id_to_dict(i)
				pool.append(plan)
				if len(pool) >= self.embedding_batch_size:
					new_embeds = self._embedding(pool)
					embeds.append(new_embeds)
					pool = []
			if len(pool) > 0:
				new_embeds = self._embedding(pool)
				embeds.append(new_embeds)
		else:

			with diskcache.Cache(self.embed_cache_path, size_limit=self.cache_size_limit) as cache:
				pool = []
				for i in trange(self.num_tasks, desc="Embedding tasks"):
					plan = self._plan_id_to_dict(i)
					key = json.dumps(plan, sort_keys=True)
					embed = None if self.overwrite_embedding_cache else cache.get(key, None)
					if embed is None:
						pool.append(plan)
						if len(pool) >= self.embedding_batch_size:
							new_embeds = self._embedding(pool)
							embeds.append(new_embeds)
							for plan, embed in zip(pool, new_embeds):
								cache.set(json.dumps(plan, sort_keys=True), embed)
							pool = []
					else:
						if len(pool) > 0:
							new_embeds = self._embedding(pool)
							embeds.append(new_embeds)
							for plan, embed in zip(pool, new_embeds):
								cache.set(json.dumps(plan, sort_keys=True), embed)
							pool = []
						embeds.append(embed)
				if len(pool) > 0:
					new_embeds = self._embedding(pool)
					embeds.append(new_embeds)
					for plan, embed in zip(pool, new_embeds):
						cache.set(json.dumps(plan, sort_keys=True), embed)
		return np.vstack(embeds)

	def get_tasks(self, indices):
		"(Abstract method) get task"

	def _plan_id_to_dict(self, plan_id):
		plan_id = int(plan_id)
		task_plan = self.task_plan_df.iloc[plan_id].dropna().to_dict()
		for k, v in task_plan.items():
			if isinstance(v, np.ndarray):
				task_plan[k] = v.tolist()
		return task_plan

	def _evaluate_one(self, plan_id, model: Union[Model, List[Model]], aggregate_func: Union[str, Callable] = None):
		"(Abstract method) evaluate task"

	def _evaluate_many(self, indices, model: Union[Model, List[Model]], aggregate_func: Union[str, Callable] = None):
		return [self._evaluate_one(i, model, aggregate_func) for i in tqdm(indices, 'Evaluating tasks')]

	def evaluate(self, model: Union[Model, List[Model]], aggregate_func: Union[str, Callable] = None):
		return self._evaluate_many(range(len(self.task_plan_df)), model, aggregate_func)

	def _get_ground_truth(
			self,
			x_indices,
			model: Union[Model, List[Model]] = None,
			ground_truth: np.ndarray = None,
			aggregate_func: Union[str, Callable] = None
	):
		assert model is not None or ground_truth is not None
		if ground_truth is None:
			ground_truth = np.array(self._evaluate_many(x_indices, model, aggregate_func))
		elif aggregate_func is not None:
			ground_truth = np.array([apply_aggregate_function(x, aggregate_func) for x in ground_truth])
		assert ground_truth.shape == (len(x_indices),)
		return ground_truth

	def _get_ground_truth_for_all(
			self,
			x_indices,
			fit: bool = False,
			approximator_model: str = 'regressor',
			function_approximator=None,
			model: Union[Model, List[Model]] = None,
			ground_truth: np.ndarray = None,
			aggregate_func: Union[str, Callable] = None
	):
		ground_truth = self._get_ground_truth(x_indices, model, ground_truth, aggregate_func)

		if fit:
			function_approximator = self.fit(
				x_indices=x_indices,
				approximator_model=approximator_model,
				function_approximator=function_approximator,
				ground_truth=ground_truth
			)
			y = function_approximator.predict(self.embeddings)
		else:
			y = np.full(self.num_tasks, np.nan)
		y[x_indices] = ground_truth
		return y

	def fit(
			self,
			x_indices,
			approximator_model: str = 'regressor',
			function_approximator=None,
			model: Union[Model, List[Model]] = None,
			ground_truth: np.ndarray = None,
			aggregate_func: Union[str, Callable] = None,
			return_ground_truth=False
	):
		ground_truth = self._get_ground_truth(x_indices, model, ground_truth, aggregate_func)
		self.check_embedding()

		if function_approximator is None:
			assert approximator_model in ['regressor', 'classifier']
			from sklearn.model_selection import GridSearchCV
			from sklearn.gaussian_process import kernels
			param_grid = {
				"alpha": np.logspace(-5, 0, 20),
			}
			if approximator_model == 'regressor':
				function_approximator = GridSearchCV(
					estimator=GaussianProcessRegressor(
						kernel=kernels.RBF(),
						normalize_y=True,
						random_state=42,
					),
					param_grid=param_grid,
					cv=5,
					n_jobs=-1
				).fit(self.embeddings[x_indices], ground_truth)
			if approximator_model == 'classifier':
				from sklearn.gaussian_process import GaussianProcessClassifier
				function_approximator = GridSearchCV(
					estimator=GaussianProcessClassifier(
						kernel=kernels.RBF(),
						random_state=42,
					),
					param_grid=param_grid,
					cv=5,
					n_jobs=-1
				).fit(self.embeddings[x_indices], ground_truth)
		else:
			function_approximator = function_approximator.fit(self.embeddings[x_indices], ground_truth)

		if return_ground_truth:
			return function_approximator, ground_truth
		else:
			return function_approximator

	def active_fit(
			self,
			warmup_budget: int,
			budget: int,
			query_func: Callable,
			model: Union[Model, List[Model]] = None,
			approximator_model: str = 'regressor',
			function_approximator=None,
			batch_size: int = 1,
			aggregate_func: Union[str, Callable] = None,
			ground_truth: np.ndarray = None,
			return_ground_truth: bool = False
	):
		assert model is not None or ground_truth is not None

		assert warmup_budget + budget < self.num_tasks, "Warmup budget + budget should be less than the number of tasks"

		print(f"[WARMUP] Querying {warmup_budget} tasks")
		x_indices = self.rng.choice(self.num_tasks, warmup_budget, replace=False)
		if ground_truth is None:
			y = np.array(self._evaluate_many(x_indices, model, aggregate_func))
		else:
			y = self._get_ground_truth(x_indices, model, ground_truth[x_indices], aggregate_func)

		function_approximator = self.fit(
			x_indices=x_indices,
			approximator_model=approximator_model,
			function_approximator=function_approximator,
			ground_truth=y
		)

		while len(y) < budget + warmup_budget:
			query_indices = query_func(
				task_evaluator=self,
				pool=self.embeddings,
				selected_indices=x_indices,
				function_approximator=function_approximator,
				batch_size=batch_size
			)
			if ground_truth is None:
				query_y = np.array(self._evaluate_many(query_indices, model, aggregate_func))
			else:
				query_y = self._get_ground_truth(query_indices, model, ground_truth[query_indices], aggregate_func)

			x_indices = np.concatenate([x_indices, query_indices])
			y = np.concatenate([y, query_y])

			function_approximator = self.fit(
				x_indices=x_indices,
				approximator_model=approximator_model,
				function_approximator=function_approximator,
				ground_truth=y
			)

			print(f"[Query] Queried {len(y) - warmup_budget}/{budget} tasks")

		if return_ground_truth:
			return function_approximator, (x_indices, y)
		else:
			return function_approximator

	def _groupby(self, by):
		if by is None:
			return [(i, np.array([i])) for i in range(self.num_tasks)]
		else:
			indices = list(self.task_plan_df.groupby(by).indices.items())
			assert len(indices) > 1, "Groupby should have more than 1 group"
			return indices

	def top_k_query(
			self,
			k: int,
			x_indices,
			model: Union[Model, List[Model]] = None,
			reverse=False,
			by=None,  # pandas groupby
			ground_truth: np.ndarray = None,
			aggregate_func: Union[str, Callable] = None,
			function_approximator=None,
			fit_function_approximator: bool = False,
	):
		indices = self._groupby(by)

		y = self._get_ground_truth_for_all(
			x_indices=x_indices,
			fit=fit_function_approximator,
			approximator_model='regressor',
			function_approximator=function_approximator,
			model=model,
			ground_truth=ground_truth,
			aggregate_func=aggregate_func
		)

		aggregate_perf = np.array([np.nanmean(y[i]) for k, i in indices])

		if reverse:
			perf_ranking = np.argsort(aggregate_perf)
		else:
			perf_ranking = np.argsort(-aggregate_perf)

		top_k = [(indices[i][0], aggregate_perf[i]) for i in perf_ranking[:k]]

		if fit_function_approximator:
			return top_k, function_approximator
		else:
			return top_k

	def active_top_k_query(
			self,
			k: int,
			warmup_budget: int,
			budget: int,
			model: Union[Model, List[Model]] = None,
			batch_size: int = 10,
			reverse=False,
			by=None,  # pandas groupby
			function_approximator=None,
			aggregate_func: Union[str, Callable] = None,
			ground_truth: np.ndarray = None,
	):
		assert model is not None or ground_truth is not None

		if warmup_budget + budget >= self.num_tasks:
			print("[IMPORTANT] Warmup budget + budget larger than the number of tasks, switch to normal query")
			return self.top_k_query(
				k=k,
				x_indices=range(self.num_tasks),
				model=model,
				reverse=reverse,
				by=by,
				ground_truth=ground_truth,
				aggregate_func=aggregate_func,
				function_approximator=function_approximator,
				fit_function_approximator=False
			), None

		self.check_embedding()
		indices = self._groupby(by)

		def query_func(
				task_evaluator,
				selected_indices,
				function_approximator,
				batch_size,
				**kwargs
		):
			pred_perf = function_approximator.predict(task_evaluator.embeddings)
			aggregate_perf = np.array([np.mean(pred_perf[i]) for k, i in indices])

			if reverse:
				perf_ranking = np.argsort(aggregate_perf)
			else:
				perf_ranking = np.argsort(-aggregate_perf)

			selected_ = []
			for i in perf_ranking:
				candidates = [ii for ii in indices[i][1] if ii not in selected_indices]
				if len(candidates) > 0:
					selected_.append(task_evaluator.rng.choice(candidates))
				if len(selected_) >= batch_size:
					break
			return np.array(selected_)

		function_approximator, (x_indices, y) = self.active_fit(
			warmup_budget=warmup_budget,
			budget=budget,
			query_func=query_func,
			model=model,
			approximator_model='regressor',
			function_approximator=function_approximator,
			batch_size=batch_size,
			return_ground_truth=True,
			aggregate_func=aggregate_func,
			ground_truth=ground_truth
		)

		pred_perf = function_approximator.predict(self.embeddings)
		pred_perf[x_indices] = y
		aggregate_perf = np.array([np.mean(pred_perf[i]) for k, i in indices])

		if reverse:
			perf_ranking = np.argsort(aggregate_perf)
		else:
			perf_ranking = np.argsort(-aggregate_perf)

		top_k = [(indices[i][0], aggregate_perf[i]) for i in perf_ranking[:k]]

		return top_k, function_approximator

	def threshold_query(
			self,
			threshold: float,
			x_indices,
			model: Union[Model, List[Model]] = None,
			greater_than=True,
			by=None,  # pandas groupby
			ground_truth: np.ndarray = None,
			aggregate_func: Union[str, Callable] = None,
			function_approximator=None,
			fit_function_approximator: bool = False,
	):
		indices = self._groupby(by)

		y = self._get_ground_truth_for_all(
			x_indices=x_indices,
			fit=fit_function_approximator,
			approximator_model='regressor',
			function_approximator=function_approximator,
			model=model,
			ground_truth=ground_truth,
			aggregate_func=aggregate_func
		)

		aggregate_perf = np.array([np.nanmean(y[i]) for k, i in indices])

		if greater_than:
			selection = np.where(aggregate_perf > threshold)[0]
		else:
			selection = np.where(aggregate_perf < threshold)[0]

		results = [(indices[i][0], aggregate_perf[i]) for i in selection]

		if fit_function_approximator:
			return results, function_approximator
		else:
			return results

	def active_threshold_query(
			self,
			threshold: float,
			warmup_budget: int,
			budget: int,
			model: Union[Model, List[Model]] = None,
			batch_size: int = 10,
			greater_than=True,
			by=None,  # pandas groupby
			function_approximator=None,
			aggregate_func: Union[str, Callable] = None,
			ground_truth: np.ndarray = None,
	):
		assert model is not None or ground_truth is not None

		if warmup_budget + budget >= self.num_tasks:
			print("[IMPORTANT] Warmup budget + budget larger than the number of tasks, switch to normal query")
			return self.threshold_query(
				threshold=threshold,
				x_indices=range(self.num_tasks),
				model=model,
				greater_than=greater_than,
				by=by,
				ground_truth=ground_truth,
				aggregate_func=aggregate_func,
				function_approximator=function_approximator,
				fit_function_approximator=False
			), None

		self.check_embedding()
		indices = self._groupby(by)

		def query_func(
				task_evaluator,
				selected_indices,
				function_approximator,
				batch_size,
				**kwargs
		):
			pred_perf = function_approximator.predict(task_evaluator.embeddings)
			aggregate_perf = np.array([np.mean(pred_perf[i]) for k, i in indices])

			abs_diff_ranking = np.argsort(np.abs(aggregate_perf - threshold))

			selected_ = []
			for i in abs_diff_ranking:
				candidates = [ii for ii in indices[i][1] if ii not in selected_indices]
				if len(candidates) > 0:
					selected_.append(task_evaluator.rng.choice(candidates))
				if len(selected_) >= batch_size:
					break
			return np.array(selected_)

		function_approximator, (x_indices, y) = self.active_fit(
			warmup_budget=warmup_budget,
			budget=budget,
			query_func=query_func,
			model=model,
			approximator_model='regressor',
			function_approximator=function_approximator,
			batch_size=batch_size,
			return_ground_truth=True,
			aggregate_func=aggregate_func,
			ground_truth=ground_truth
		)

		pred_perf = function_approximator.predict(self.embeddings)
		pred_perf[x_indices] = y
		aggregate_perf = np.array([np.mean(pred_perf[i]) for k, i in indices])

		if greater_than:
			selection = np.where(aggregate_perf > threshold)[0]
		else:
			selection = np.where(aggregate_perf < threshold)[0]

		results = [(indices[i][0], aggregate_perf[i]) for i in selection]

		return results, function_approximator

	def model_debug(
			self,
			x_indices,
			model: Model = None,
			k: int = 10,
			greater_than=True,
			threshold: float = None,
			ground_truth: np.ndarray = None,
			function_approximator=None,
			fit_function_approximator: bool = False,
	):

		if threshold is None:
			ground_truth = self._get_ground_truth(x_indices, model, ground_truth)
			perf_mean, perf_std = np.nanmean(ground_truth), np.nanstd(ground_truth)
			threshold = perf_mean + perf_std if greater_than else perf_mean - perf_std

		if fit_function_approximator:
			results, function_approximator = self.threshold_query(
				threshold=threshold,
				x_indices=x_indices,
				model=model,
				greater_than=greater_than,
				ground_truth=ground_truth,
				function_approximator=function_approximator,
				fit_function_approximator=True
			)
		else:
			results = self.threshold_query(
				threshold=threshold,
				x_indices=x_indices,
				model=model,
				greater_than=greater_than,
				ground_truth=ground_truth,
				function_approximator=function_approximator,
				fit_function_approximator=False
			)

		selection = [i for i, _ in results]
		results = find_frequent_patterns(k, self.task_plan_df.iloc[selection])

		if fit_function_approximator:
			return results, selection, function_approximator
		else:
			return results, selection

	def active_model_debug(
			self,
			warmup_budget: int,
			budget: int,
			model: Model = None,
			k: int = 10,
			greater_than=True,
			threshold: float = None,
			ground_truth: np.ndarray = None,
			function_approximator=None,
			batch_size: int = 10,
	):

		assert model is not None or ground_truth is not None

		if warmup_budget + budget >= self.num_tasks:
			print("[IMPORTANT] Warmup budget + budget larger than the number of tasks, switch to normal query")
			results, selection = self.model_debug(
				x_indices=range(self.num_tasks),
				model=model,
				k=k,
				greater_than=greater_than,
				threshold=threshold,
				ground_truth=ground_truth,
				function_approximator=function_approximator,
				fit_function_approximator=False
			)
			return results, selection, None

		self.check_embedding()

		print(f"[WARMUP] Querying {warmup_budget} tasks")
		x_indices = self.rng.choice(self.num_tasks, warmup_budget, replace=False)
		if ground_truth is None:
			y = np.array(self._evaluate_many(x_indices, model))
		else:
			y = self._get_ground_truth(x_indices, model, ground_truth[x_indices])

		function_approximator = self.fit(
			x_indices=x_indices,
			approximator_model='regressor',
			function_approximator=function_approximator,
			ground_truth=y
		)

		if threshold is None:
			perf_mean, perf_std = np.nanmean(y), np.nanstd(y)
			threshold = perf_mean + perf_std if greater_than else perf_mean - perf_std

		def query_func(
				pool,
				selected_indices,
				function_approximator,
				batch_size,
				**kwargs
		):
			pred_perf = function_approximator.predict(pool)
			abs_diff_ranking = np.argsort(np.abs(pred_perf - threshold))

			selected_ = []
			for i in abs_diff_ranking:
				if i not in selected_indices:
					selected_.append(i)
					if len(selected_) >= batch_size:
						break
			return np.array(selected_)

		while len(y) < budget + warmup_budget:
			query_indices = query_func(
				pool=self.embeddings,
				selected_indices=x_indices,
				function_approximator=function_approximator,
				batch_size=batch_size
			)

			if ground_truth is None:
				query_y = np.array(self._evaluate_many(query_indices, model))
			else:
				query_y = self._get_ground_truth(query_indices, model, ground_truth[query_indices])

			x_indices = np.concatenate([x_indices, query_indices])
			y = np.concatenate([y, query_y])

			function_approximator = self.fit(
				x_indices=x_indices,
				approximator_model='regressor',
				function_approximator=function_approximator,
				ground_truth=y
			)

			print(f"[Query] Queried {len(y) - warmup_budget}/{budget} tasks")

		pred_perf = function_approximator.predict(self.embeddings)
		pred_perf[x_indices] = y

		if greater_than:
			selection = np.where(pred_perf > threshold)[0]
		else:
			selection = np.where(pred_perf < threshold)[0]

		results = find_frequent_patterns(k, self.task_plan_df.iloc[selection])

		return results, selection, function_approximator

	def model_compare(
			self,
			x_indices,
			model: Model = None,
			baselines: Union[Model, List[Model]] = None,
			k: int = 10,
			threshold: float = 0.0,
			greater_than=True,
			ground_truth: np.ndarray = None,
			function_approximator=None,
			fit_function_approximator: bool = False,
	):
		if model is None:
			models = None
		else:
			models = [model] + baselines

		def aggregate_func(x):
			if greater_than:
				return x[0] - np.max(x[1:]) - threshold
			else:
				return x[0] - np.min(x[1:]) + threshold

		if fit_function_approximator:
			results, function_approximator = self.threshold_query(
				threshold=0,
				x_indices=x_indices,
				model=models,
				greater_than=greater_than,
				ground_truth=ground_truth,
				function_approximator=function_approximator,
				fit_function_approximator=True,
				aggregate_func=aggregate_func
			)
		else:
			results = self.threshold_query(
				threshold=0,
				x_indices=x_indices,
				model=models,
				greater_than=greater_than,
				ground_truth=ground_truth,
				function_approximator=function_approximator,
				fit_function_approximator=False,
				aggregate_func=aggregate_func
			)

		selection = [i for i, _ in results]
		results = find_frequent_patterns(k, self.task_plan_df.iloc[selection])

		if fit_function_approximator:
			return results, selection, function_approximator
		else:
			return results, selection

	def active_model_compare(
			self,
			warmup_budget: int,
			budget: int,
			model: Model = None,
			baselines: Union[Model, List[Model]] = None,
			batch_size: int = 10,
			k: int = 10,
			threshold: float = 0.0,
			greater_than=True,
			function_approximator=None,
			ground_truth: np.ndarray = None,
	):
		if model is None:
			models = None
		else:
			models = [model] + baselines

		def aggregate_func(x):
			if greater_than:
				return x[0] - np.max(x[1:]) - threshold
			else:
				return x[0] - np.min(x[1:]) + threshold

		results, function_approximator = self.active_threshold_query(
			threshold=0,
			warmup_budget=warmup_budget,
			budget=budget,
			model=models,
			batch_size=batch_size,
			greater_than=greater_than,
			function_approximator=function_approximator,
			aggregate_func=aggregate_func,
			ground_truth=ground_truth,
		)

		selection = [i for i, _ in results]
		results = find_frequent_patterns(k, self.task_plan_df.iloc[selection])

		return results, selection, function_approximator


class QATaskEvaluator(TaskEvaluator):
	data_field = None

	def __init__(
			self,
			task_plan_df,
			task_generator,
			embedding_func: Callable = None,
			embedding_name: str = 'st',
			embedding_batch_size: int = 10000,
			n_instance_per_task: int = 5,
			n_trials_per_instance: int = 3,
			cache_path_root: str = None,
			cache_size_limit: int = 10,  # in GB
			overwrite_embedding_cache: bool = False,
			overwrite_eval_cache: bool = False,
			overwrite_task_cache: bool = False,
			seed: int = 42,
	):
		super().__init__(
			task_plan_df,
			task_generator,
			embedding_func,
			embedding_name,
			embedding_batch_size,
			cache_path_root,
			cache_size_limit,
			overwrite_embedding_cache,
			overwrite_eval_cache,
			overwrite_task_cache,
			seed,
		)
		self.n_instance_per_task = n_instance_per_task
		self.n_trials_per_instance = n_trials_per_instance

	def _embedding(self, pool):
		docs = []
		for plan in pool:
			task = self.task_generator.generate(plan, return_data=False)
			docs.append(f'question: {task["question"]}\nanswer: {task["answer"]}\n{task["task_plan"]}')
		return self.embedding_func(docs)

	def _generate_task(self, task_plan, i):
		if self.task_cache_path is None:
			task = self.task_generator.generate(task_plan, return_data=True)
			task['options_trials'] = [list(self.rng.permutation(task['options'])) for _ in range(self.n_trials_per_instance)]
		else:
			with diskcache.Cache(self.task_cache_path, size_limit=self.cache_size_limit) as cache:
				key = task_plan.copy()
				key['instance_id'] = i
				key_str = json.dumps(key, sort_keys=True)
				task = None if self.overwrite_task_cache else cache.get(key_str, None)
				if task is None:
					task = self.task_generator.generate(task_plan, return_data=True)
					task['options_trials'] = [list(self.rng.permutation(task['options'])) for _ in range(self.n_trials_per_instance)]
				cache.set(key_str, task)

		return task

	def get_tasks(self, indices):
		tasks = {}
		for plan_id in indices:
			plan_id = int(plan_id)
			task_plan = self._plan_id_to_dict(plan_id)
			tasks[plan_id] = {}
			for i in range(self.n_instance_per_task):
				tasks[plan_id][i] = self._generate_task(task_plan, i)

		return tasks

	def _evaluate_task_plan(self, task_plan, model: QAModel):
		acc = []
		for i in range(self.n_instance_per_task):
			task = self._generate_task(task_plan, i)
			for ii in range(self.n_trials_per_instance):
				res = model.multiple_choice_qa(task[self.data_field], task['question'], task['options_trials'][ii], task['answer'])
				acc.append(res['accuracy'])

		acc = np.mean(acc)
		return acc

	def _evaluate_one(self, plan_id, model: Union[QAModel, List[QAModel]], aggregate_func: Union[str, Callable] = None):
		plan_id = int(plan_id)
		task_plan = self._plan_id_to_dict(plan_id)

		if self.eval_cache_path is None:
			if isinstance(model, list):
				acc = [self._evaluate_task_plan(task_plan, m) for m in model]
			else:
				acc = self._evaluate_task_plan(task_plan, model)
		else:
			with diskcache.Cache(self.eval_cache_path, size_limit=self.cache_size_limit) as cache:
				key = task_plan.copy()
				key['n_trials_per_instance'] = self.n_trials_per_instance
				key['n_instance_per_task'] = self.n_instance_per_task

				if isinstance(model, list):
					acc = []
					for m in model:
						key['model'] = m.model_name
						key_str = json.dumps(key, sort_keys=True)
						acc_m = None if self.overwrite_eval_cache else cache.get(key_str, None)
						if acc_m is None:
							acc_m = self._evaluate_task_plan(task_plan, m)
							cache.set(key_str, acc_m)
						acc.append(acc_m)
				else:
					key['model'] = model.model_name
					key_str = json.dumps(key, sort_keys=True)
					acc = None if self.overwrite_eval_cache else cache.get(key_str, None)
					if acc is None:
						acc = self._evaluate_task_plan(task_plan, model)
						cache.set(key_str, acc)
		res = apply_aggregate_function(acc, aggregate_func)
		return res


class VQATaskEvaluator(QATaskEvaluator):
	data_field = 'image'


class VideoQATaskEvaluator(QATaskEvaluator):
	data_field = 'video'
