from setuptools import find_packages, setup
from typing import List

HYPEN_DOT = '-e .'
def getRequirements(filePath : str) -> List[str]:
    ''' This function will return the list of requirements.txt '''
    reqs = []
    with open(filePath, encoding='utf-8', mode='r+') as file:
        lines = file.readlines()
        lines = [line.replace('/n', '') for line in lines]
        reqs = lines
    
    if HYPEN_DOT in reqs:
        reqs.remove(HYPEN_DOT)
    
    return reqs
    

setup(
    name='end2endMLProject',
    version='1.0.0',
    author='Karthikeyan Anumalla',
    author_email='anumallakarthikeyan03@gmail.com',
    packages=find_packages(),
    install_requires = getRequirements('./requirements.txt')
)

