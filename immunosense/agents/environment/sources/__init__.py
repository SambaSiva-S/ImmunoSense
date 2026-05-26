"""Environmental data sources.

Each source implements the EnvironmentDataSource Protocol:
    fetch(latitude, longitude, target_date) -> FetchedFeatures
"""

from immunosense.agents.environment.sources.airnow import AirNowSource
from immunosense.agents.environment.sources.composite import CompositeEnvironmentSource
from immunosense.agents.environment.sources.google_pollen import GooglePollenSource
from immunosense.agents.environment.sources.mock import MockEnvironmentSource
from immunosense.agents.environment.sources.openweather import OpenWeatherSource

__all__ = [
    "AirNowSource",
    "CompositeEnvironmentSource",
    "GooglePollenSource",
    "MockEnvironmentSource",
    "OpenWeatherSource",
]
