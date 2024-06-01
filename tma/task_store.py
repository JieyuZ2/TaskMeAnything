import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa_schema_map = {
	'str' : pa.string(),
	'int' : pa.int64(),
	'list': pa.list_(pa.string()),
}

pd_schema_map = {
	'str' : 'string',
	'int' : 'Int64',
	'list': 'object',
}


def get_pa_schema(schema):
	return pa.schema([(k, pa_schema_map[v]) for k, v in schema.items()])


def get_pd_schema(schema):
	return {k: pd_schema_map[v] for k, v in schema.items()}


class TaskStore:

	def __init__(self, schema, output_file=None, buffer_size=1e8):
		self.columns = list(schema.keys())
		self.dtypes = list(schema.values())
		self.buffer = []
		self.buffer_size = buffer_size
		self.output_file = output_file
		if output_file is None:
			self.schema = get_pd_schema(schema)
			self.task_plan_df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in self.schema.items()})
		else:
			print(f'Writing to {output_file}')
			self.counter = 0
			self.schema = get_pa_schema(schema)
			self.parquet_writer = pq.ParquetWriter(output_file, schema=self.schema)

	def _update_buffer(self):
		if len(self.buffer) > self.buffer_size:
			self.dump()

	def dump(self):
		if len(self.buffer) > 0:
			if self.output_file is None:
				self.task_plan_df = pd.concat(
					[self.task_plan_df, pd.DataFrame(self.buffer, columns=self.columns).astype(self.schema, errors='ignore')],
					ignore_index=True,
					sort=False
				)
			else:
				self.parquet_writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
				self.counter += len(self.buffer)
			self.buffer = []

	def add_many(self, xs):
		self.buffer.extend(xs)
		self._update_buffer()

	def add(self, x):
		self.buffer.append(x)
		self._update_buffer()

	def __len__(self):
		if self.output_file is None:
			return len(self.task_plan_df) + len(self.buffer)
		else:
			return self.counter + len(self.buffer)

	def return_df(self):
		self.dump()
		return self.task_plan_df

	def close(self):
		if self.output_file is not None:
			self.dump()
			self.parquet_writer.close()
