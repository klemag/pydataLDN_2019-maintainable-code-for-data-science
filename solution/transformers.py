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
        return self

    def transform(self, X):
        categories = X["category"]

        return pd.DataFrame({
            "gen_cat": categories.apply(lambda x: self._get_slug(x)[0]),
            "precise_cat": categories.apply(lambda x: self._get_slug(x)[1])
        })


class GoalAdjustor(BaseEstimator, TransformerMixin):
    """Adjusts the goal feature to USD"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"adjusted_goal": X.goal * X.static_usd_rate})


class TimeTransformer(BaseEstimator, TransformerMixin):
    """Builds features computed from timestamps"""

    adj = 1_000_000_000

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        deadline = pd.to_datetime(X.deadline * self.adj)
        created = pd.to_datetime(X.created_at * self.adj)
        launched = pd.to_datetime(X.launched_at * self.adj)

        return pd.DataFrame({
            "launched_to_deadline": (deadline - launched).dt.days,
            "created_to_launched": (launched - created).dt.days
        })


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
        return self

    def transform(self, X):
        return pd.DataFrame({"country": X.country.map(self.countries)})
