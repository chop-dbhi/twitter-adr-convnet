__author__ = 'Aaron J. Masino'

import ConfigParser

def loadConfig(path):
    config = ConfigParser.ConfigParser()
    config.read(path)
    return config

def ConfigSectionMap(config, section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option.upper()] = config.get(section, option)
            if dict1[option.upper()] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option.upper())
            dict1[option.upper()] = None
    return dict1

def getListInt(config, section, option, delim=','):
    return [int(x) for x in config.get(section,option).split(delim)]

def getListFloat(config, section, option, delim=','):
    return [float(x) for x in config.get(section,option).split(delim)]

def getListBool(config, section, option, delim=','):
    return [x.lower().strip()=='true' for x in config.get(section,option).split(delim)]