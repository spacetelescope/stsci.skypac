"""
This module provides information about supported HST instruments for use
by :py:mod:`stsci.skypac` module.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: `<http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE>`_

"""
__version__ = '0.1'
__vdate__ = '11-Jul-2014'
__author__ = 'Mihai Cara'

supported_telescopes    = [ 'HST' ]

counts_only_instruments = [ 'WFPC','WFPC2','ACS', 'STIS' ] # 'HRS'
mixed_units_instruments = [ 'NICMOS','WFC3' ]
rates_only_instruments  = [ 'FOC','COS' ]

supported_instruments = counts_only_instruments+mixed_units_instruments+ \
                        rates_only_instruments

photcorr_kwd = {
    'FOC'    : [ 'WAVCORR',  'COMPLETE'  ],
    'WFPC'   : [ 'DOPHOTOM', 'DONE'      ],
    'WFPC2'  : [ 'DOPHOTOM', 'COMPLETE'  ],
    'NICMOS' : [ 'PHOTDONE', 'PERFORMED' ],
    'STIS'   : [ 'PHOTCORR', 'COMPLETE'  ],
    'ACS'    : [ 'PHOTCORR', 'COMPLETE'  ],
    'WFC3'   : [ 'PHOTCORR', 'COMPLETE'  ],
    'COS'    : [ 'PHOTCORR', 'COMPLETE'  ]
}
