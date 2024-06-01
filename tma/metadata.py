from typing import List

import networkx as nx


class MetaData:
	"""
	Abstract class for metadata
	"""


class CategoryMetaData(MetaData):
	def __init__(self):
		super().__init__()

		self.taxonomy = None
		self.categories = None
		self.category_info = None

	def check_category_exists(self, cateid):
		return cateid in self.categories

	def get_surfacename(self, node):
		return self.category_info[node]['surface_name'][0]

	def get_relevant_categories(self, cateid):
		return set(nx.descendants(self.taxonomy, cateid)) | set(nx.ancestors(self.taxonomy, cateid)) | {cateid}

	def get_irrelevant_categories(self, cateid):
		if isinstance(cateid, List):
			relevant_categories = set()
			for c in cateid:
				relevant_categories |= self.get_relevant_categories(c)
		else:
			relevant_categories = self.get_relevant_categories(cateid)
		return set(self.categories) - relevant_categories
