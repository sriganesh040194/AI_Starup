import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace("c51cd33f-083e-4304-a95b-442a52dc4a2a", 
               "NetworkWatcherRG", "eSpaceAI", auth=None, _location=None, _disable_service_check=False)