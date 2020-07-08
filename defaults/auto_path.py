#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2018-2019, Zhang Yunjun                     #
# Author:  Zhang Yunjun, 2018 Mar                          #
# Modified by: Sara Mirzaee                                #
############################################################


import os
import re
import glob
import numpy as np
# Auto setting for file structure of Univ. of Miami, as shown below.
# It required 3 conditions: 1) autoPath = True
#                           2) $SCRATCHDIR is defined in environmental variable
#                           3) input custom template with basename same as project_name
# Change it to False if you are not using it.
autoPath = True


# Default path of data files from different InSAR processors to be loaded into MintPy
isceTopsAutoPath = '''##----------Default file path of ISCE/topsStack products
minopy.load.processor      = isce
minopy.load.metaFile       = ${PROJECT_DIR}/master/IW*.xml
minopy.load.baselineDir    = ${PROJECT_DIR}/baselines

minopy.load.slcFile        = ${PROJECT_DIR}/merged/SLC/*/*.slc.full
minopy.load.unwFile        = ${PROJECT_DIR}/minopy/inverted/interferograms/*/fine*.unw
minopy.load.corFile        = ${PROJECT_DIR}/minopy/inverted/interferograms/*/fine*.cor
minopy.load.connCompFile   = ${PROJECT_DIR}/minopy/inverted/interferograms/*/fine*.unw.conncomp
minopy.load.ionoFile       = None
minopy.load.intFile        = None

minopy.load.demFile        = ${PROJECT_DIR}/merged/geom_master/hgt.rdr.full
minopy.load.lookupYFile    = ${PROJECT_DIR}/merged/geom_master/lat.rdr.full
minopy.load.lookupXFile    = ${PROJECT_DIR}/merged/geom_master/lon.rdr.full
minopy.load.incAngleFile   = ${PROJECT_DIR}/merged/geom_master/los.rdr.full
minopy.load.azAngleFile    = ${PROJECT_DIR}/merged/geom_master/los.rdr.full
minopy.load.shadowMaskFile = ${PROJECT_DIR}/merged/geom_master/shadowMask.rdr.full
minopy.load.bperpFile      = None

'''

isceStripmapAutoPath = '''##----------Default file path of ISCE/stripmapStack products
minopy.load.processor      = isce
minopy.load.metaFile       = ${masterShelve}/masterShelve/data.dat
minopy.load.baselineDir    = ${PROJECT_DIR}/baselines

minopy.load.slcFile        = ${PROJECT_DIR}/merged/SLC/*/*.slc
minopy.load.unwFile        = ${PROJECT_DIR}/minopy/inverted/interferograms/*/fine*.unw
minopy.load.corFile        = ${PROJECT_DIR}/minopy/inverted/interferograms/*/fine*.cor
minopy.load.connCompFile   = ${PROJECT_DIR}/minopy/inverted/interferograms/*/fine*.unw.conncomp
minopy.load.ionoFile       = None
minopy.load.intFile        = None

minopy.load.demFile        = ${PROJECT_DIR}/merged/geom_master/hgt.rdr.full
minopy.load.lookupYFile    = ${PROJECT_DIR}/merged/geom_master/lat.rdr.full
minopy.load.lookupXFile    = ${PROJECT_DIR}/merged/geom_master/lon.rdr.full
minopy.load.incAngleFile   = ${PROJECT_DIR}/merged/geom_master/los.rdr.full
minopy.load.azAngleFile    = ${PROJECT_DIR}/merged/geom_master/los.rdr.full
minopy.load.shadowMaskFile = ${PROJECT_DIR}/merged/geom_master/shadowMask.rdr.full
minopy.load.bperpFile      = None

'''

roipacAutoPath = '''##----------Default file path of ROI_PAC products
minopy.load.processor      = roipac
minopy.load.unwFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/filt*.unw
minopy.load.corFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/filt*.cor
minopy.load.connCompFile   = ${PROJECT_DIR}/PROCESS/DONE/IFG*/filt*snap_connect.byt
minopy.load.intFile        = None

minopy.load.demFile        = ${PROJECT_DIR}/PROCESS/DONE/*${m_date12}*/radar_*rlks.hgt
minopy.load.lookupYFile    = ${PROJECT_DIR}/PROCESS/GEO/geo_${m_date12}/geomap_*rlks.trans
minopy.load.lookupXFile    = ${PROJECT_DIR}/PROCESS/GEO/geo_${m_date12}/geomap_*rlks.trans
minopy.load.incAngleFile   = None
minopy.load.azAngleFile    = None
minopy.load.shadowMaskFile = None
minopy.load.bperpFile      = None
'''

gammaAutoPath = '''##----------Default file path of GAMMA products
minopy.load.processor      = gamma
minopy.load.unwFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/diff*rlks.unw
minopy.load.corFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/*filt*rlks.cor
minopy.load.connCompFile   = None
minopy.load.intFile        = None

minopy.load.demFile        = ${PROJECT_DIR}/PROCESS/SIM/sim_${m_date12}/sim*.hgt_sim
minopy.load.lookupYFile    = ${PROJECT_DIR}/PROCESS/SIM/sim_${m_date12}/sim*.UTM_TO_RDC
minopy.load.lookupXFile    = ${PROJECT_DIR}/PROCESS/SIM/sim_${m_date12}/sim*.UTM_TO_RDC
minopy.load.incAngleFile   = None
minopy.load.azAngleFile    = None
minopy.load.shadowMaskFile = None
minopy.load.bperpFile      = ${PROJECT_DIR}/merged/baselines/*/*.base_perp
'''

autoPathDict = {
    'isceTops'  : isceTopsAutoPath,
    'isceStripmap'  : isceStripmapAutoPath,
    'roipac': roipacAutoPath,
    'gamma' : gammaAutoPath,
}

prefix = 'minopy.load.'


class PathFind:
    def __init__(self):
        self.scratchdir = os.getenv('SCRATCHDIR')
        self.defaultdir = os.path.expandvars('${MINOPY_HOME}/minopy/defaults')
        self.geomasterdir = 'subset/geom_master'
        #self.patchdir = 'patches'
        self.minopydir = 'minopy'
        self.rundir = 'run_files'
        self.configdir = 'configs'
        self.mergeddir = 'subset'
        self.intdir = 'inverted/interferograms'
        self.auto_template = self.defaultdir + '/minopy_template.cfg'
        self.wrappercommandtops = 'SentinelWrapper.py -c '
        self.wrappercommandstripmap = 'stripmapWrapper.py -c '
        return

##----------------------------------------------------------------------------------------##


def read_str2dict(inString, delimiter='=', print_msg=False):
    '''Read multiple lines of string into dict
    Based on mintpy.utils.readfile.read_template()
    '''
    strDict = {}
    lines = inString.split('\n')
    for line in lines:
        c = [i.strip() for i in line.strip().split(delimiter, 1)]
        if len(c) < 2 or line.startswith(('%', '#')):
            next
        else:
            key = c[0]
            value = str.replace(c[1], '\n', '').split("#")[0].strip()
            if value != '':
                strDict[key] = value

    # set 'None' to None
    for key, value in strDict.items():
        if value.lower() == 'none':
            strDict[key] = None
    return strDict


def get_master_date12(project_dir, processor='roipac'):
    """
    date12 of reference interferogram in YYMMDD-YYMMDD format
    """

    m_date12 = None

    # opt 1 - master_ifgram.txt
    m_ifg_file = os.path.join(project_dir, 'PROCESS', 'master_ifgram.txt')
    if os.path.isfile(m_ifg_file):
        m_date12 = str(np.loadtxt(m_ifg_file, dtype=bytes).astype(str))
        return m_date12

    # opt 2 - folders under GEO/SIM
    if processor == 'roipac':
        try:
            lookup_file = glob.glob(os.path.join(project_dir, 'PROCESS/GEO/geo_*/geomap*.trans'))[0]
            m_date12 = re.findall('\d{6}-\d{6}', lookup_file)[0]
        except:
            print("No master interferogram found! Check the PROCESS/GEO/geo_* folder")

    elif processor == 'gamma':
        geom_dir = os.path.join(project_dir, 'PROCESS/SIM')
        try:
            m_date12 = os.walk(geom_dir).next()[1][0].split('sim_')[1]
        except:
            print("No master interferogram found! Check the PROCESS/SIM/sim_* folder")
    return m_date12


def get_auto_path(processor, project_name, template=dict()):
    """Update template options with auto path defined in autoPathDict
    Parameters: processor : str, isce / roipac / gamma
                project_name : str, Project name, e.g. GalapagosSenDT128
                template : dict,
    Returns:    template : dict,
    """
    project_dir = os.path.join(os.getenv('SCRATCHDIR'), project_name)
    # read auto_path_dict
    if processor == 'isce':
        if os.path.exists(project_dir + '/master'):
            processor = 'isceTops'
            template['sensor_type'] = 'tops'
        else:
            processor = 'isceStripmap'
            template['sensor_type'] = 'stripmap'

    auto_path_dict = read_str2dict(autoPathDict[processor], print_msg=False)

    # grab variable value: SCRATCHDIR, m_date12

    m_date12 = None
    if processor in ['roipac', 'gamma']:
        m_date12 = get_master_date12(project_dir, processor=processor)
        if m_date12 and processor == 'roipac':
            # determine nlooks in case both radar_2rlks.hgt and radar_8rlks.hgt exist.
            lookup_file = os.path.join(project_dir, 'PROCESS/GEO/geo_{}/geomap*.trans'.format(m_date12))
            lks = re.findall('_\d+rlks', glob.glob(lookup_file)[0])[0]
            dem_file = os.path.join('${PROJECT_DIR}/PROCESS/DONE/*${m_date12}*', 'radar{}.hgt'.format(lks))
            auto_path_dict[prefix+'demFile'] = dem_file

    var_dict = {}
    var_dict['${PROJECT_DIR}'] = project_dir
    if m_date12:
        var_dict['${m_date12}'] = m_date12
    
    if processor == 'isceStripmap':
        if template['minopy.load.metaFile'] == 'auto':
            var_dict['${masterShelve}'] = os.path.join(project_dir, 'merged/SLC', os.listdir(os.path.join(project_dir, 'merged/SLC'))[0])

    # update auto_path_dict
    for key, value in auto_path_dict.items():
        if value:
            for var1, var2 in var_dict.items():
                value = value.replace(var1, var2)
            auto_path_dict[key] = value

    # update template option with auto value
    for key, value in auto_path_dict.items():
        if value and template[key] == 'auto':
            template[key] = value
    return template


