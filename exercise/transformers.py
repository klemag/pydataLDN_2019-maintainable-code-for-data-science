import json

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoriesExtractor(BaseEstimator, TransformerMixin):
    """Extract Categories from json string.

    By default it will only keep the hardcoded categories defined below
    to avoid having too many dummies."""

    misc = "misc"
    gen_cats = ["music", "film & video", "publishing", "art", "games"]
    precise_cats = [
        "rock", "fiction", "webseries", "indie rock", "children's books",
        "shorts", "documentary", "video games"
    ]

    def __init__(self, use_all=False):
        self.use_all = use_all

    def _get_slug(self, x):
        categories = json.loads(x).get("slug", "/").split("/")

        # Only keep hardcoded categories
        if not self.use_all:
            if categories[0] not in self.gen_cats:
                categories[0] = self.misc
            if categories[1] not in self.precise_cats:
                categories[1] = self.misc

        return categories

    def fit(self, X, y=None):
        # Your code here
        # [...]
        pass

    def transform(self, X):
        # Your code here
        # [...]
        pass


class GoalAdjustor(BaseEstimator, TransformerMixin):
    """Adjusts the goal feature to USD"""

    def fit(self, X, y=None):
        # Your code here
        # [...]
        pass

    def transform(self, X):
        # Your code here
        # [...]
        pass


class TimeTransformer(BaseEstimator, TransformerMixin):
    """Builds features computed from timestamps"""

    adj = 1_000_000_000

    def fit(self, X, y=None):
        # Your code here
        # [...]
        pass

    def transform(self, X):
        # Your code here
        # [...]
        pass


class CountryTransformer(BaseEstimator, TransformerMixin):
    """Transform countries into larger groups to avoid having
    too many dummies."""

    countries = {
        'US': 'US',
        'CA': 'Canada',
        'GB': 'UK & Ireland',
        'AU': 'Oceania',
        'IE': 'UK & Ireland',
        'SE': 'Europe',
        'CH': "Europe",
        'IT': 'Europe',
        'FR': 'Europe',
        'NZ': 'Oceania',
        'DE': 'Europe',
        'NL': 'Europe',
        'NO': 'Europe',
        'MX': 'Other',
        'ES': 'Europe',
        'DK': 'Europe',
        'BE': 'Europe',
        'AT': 'Europe',
        'HK': 'Other',
        'SG': 'Other',
        'LU': 'Europe'
    }

    def fit(self, X, y=None):
        # Your code here
        # [...]
        pass

    def transform(self, X):
        # Your code here
        # [...]
        pass
