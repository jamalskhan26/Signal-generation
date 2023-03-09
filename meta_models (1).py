# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 2020

@author: Jamal Khan
"""

from abc import ABCMeta, abstractmethod

class MetaModel:
	"""
	meta model class
	"""
	__metaclass__ = ABCMeta

	@abstractmethod
	def fit(self, **kwargs):
		raise NotImplementedError("fit() not executed")

	@abstractmethod
	def performance_stats(self, *args):
		raise NotImplementedError("performance_stats() not executed")

	@abstractmethod
	def construct_strategy(self, *args):
		raise NotImplementedError("construct_strategy() not executed")

	@abstractmethod
	def output(self, *args):
		raise NotImplementedError("fit() not executed")

	def _check_if_fitted(self):
		return self.models is not None

