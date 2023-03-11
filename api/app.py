import os
from argparse import Namespace
from types import MethodType
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from sqlalchemy import __version__ as sqlalchemy_version
from sqlalchemy import inspect
from sqlalchemy.sql import func
import astropy.units as u
from astropy.constants import c as lightspeed
from astropy.table import Table, MaskedColumn

# DESI software
import sys
software_path = os.environ['$DESI_SOFTWARE_PATH']
sys.path.insert(0, f'{software_path}/desiutil/master/py')
from desiutil.log import get_logger, DEBUG
sys.path.insert(0, f'{software_path}/desitarget/master/py')
from desitarget.targetmask import (desi_mask, mws_mask, bgs_mask)
# from desisim.spec_qa import redshifts as dsq_z
sys.path.insert(0, f'{software_path}/desisurvey/master/py')
from desisurvey import __version__ as desisurvey_version
from desisurvey.ephem import get_ephem, get_object_interpolator
from desisurvey.utils import get_observer
sys.path.insert(0, f'{software_path}/desispec/master/py')
from desispec import __version__ as desispec_version
import desispec.database.redshift as db

# Paths to files, etc.
specprod = os.environ['SPECPROD'] = 'fuji'
basedir = os.path.join(os.environ['DESI_SPECTRO_REDUX'], specprod)
os.environ['DESISURVEY_OUTPUT'] = os.environ['SCRATCH']
ephem = get_ephem()
from astropy.time import Time
from astropy.coordinates import ICRS
workingdir = os.getcwd()
print(workingdir)

# Database Setup
db.log = get_logger()
postgresql = db.setup_db(schema=specprod, hostname='nerscdb03.nersc.gov', username='desi')

#Flask Setup
from flask import Flask, request, jsonify
app = Flask(__name__)
if __name__ == '__main__':
    app.run(debug=False)

valid_spectypes = {'GALAXY', 'STAR', 'QSO'}
valid_subtypes = {'CV', 'M', 'G', 'K'}
default_limit = 100

def filter_query(q, db_ref, body, z_min=-1.0, z_max=6.0, spectype=None, subtype=None, limit=None, start=None, end=None):
    """
    Filters query based on options and provided reference table
    @Params:
        q (SQLAlchemy Query): Query object to apply filters
        db_ref (SQLAlchemy DeclarativeMeta): Table to use to apply filters (either Zpix or Ztile)
    
    @Returns:
        q (SQLAlchemy Query): Query object after filters have been applied
    """
    z_min = body.get('z_min', -1.0)
    z_max = body.get('z_max', 6.0)
    spectype = body.get('spectype', None)
    subtype = body.get('subtype', None)
    limit = body.get('limit', None)
    start = body.get('start', None)
    end = body.get('end', None)
    if (z_min > z_max):
        raise ValueError(f'z_min({z_min}) must be less than z_max({z_max})')
    if (spectype and spectype not in valid_spectypes):
        raise ValueError(f'Spectype {spectype} is not valid. Choose from available spectypes: {valid_spectypes}')
    
    if (subtype and subtype not in valid_subtypes):
        raise ValueError(f'Subtype {subtype} is not valid. Choose from available subtypes: {valid_subtypes}')
        
    if (spectype and subtype and spectype != 'STAR'):
        raise ValueError('Only STAR spectype currently have subtypes')
    
    q = q.filter(db_ref.z >= z_min).filter(db_ref.z <= z_max)
    if spectype:
        q = q.filter(db_ref.spectype == spectype)
    if subtype:
        q = q.filter(db_ref.subtype == subtype)
    
    count = q.count()
    
    if limit is not None:
        if start is not None and end is not None:
            raise ValueError('Cannot handle limit and start/end arguments to filter query')
        elif (start is not None and end is None):
            q = q.offset(start).limit(limit)
        elif (end is not None and start is None):
            if end-limit < 0:
                raise IndexError(f'Invalid end argument {end} for provided limit {limit}')
            else:
                q = q.offset(end-limit).limit(limit)
        else:
            q = q.limit(limit)
    else:
        if start is None and end is None:
            q.limit(default_limit)
        elif start is None or end is None:
            raise ValueError(f'Must provide both start and end parameters if limit is not provided')
        elif end <= start:
            raise ValueError(f'Start parameter {start} must be less than end parameter {end}')
        else:
            q = q.offset(start).limit(end-start)
    
    return q

def formatJSON(q):
    results = []
    for target in q.all():
        results.append(dict(target))
    return jsonify(results)

@app.route('/query/target/<targetID>', methods=['GET'])
def getRedshiftByTargetID(targetID):
    """ 
    @Params: 
        targetID (BIGINT): Big Integer representing which object to query for redshift
    
    @Returns:
        z (DOUBLE): Redshift of the first object 
    """
    if (targetID < 0):
        raise ValueError(f'Target ID {targetID} is invalid')
    
    q = db.dbSession.query(db.Zpix.z).filter(db.Zpix.targetid == targetID)
    
    if (q.first() is None):
        raise ValueError(f'Target ID {targetID} was not found')
    if (q.count() > 1):
        print(f'More than one redshift value found for target: {targetID}. Returning first found')
        
    z = q[0][0]
    return jsonify(z)


@app.route('/query/ztile/', methods=['POST'])
def getRedshiftsByTileID():
    """ 
    @Params: 
        body (DICT): Contains query parameters.
            MUST CONTAIN: tileID, (limit/start/end)
            OPTIONAL: spectype, subtype, z_min, z_max
    
    @Returns:
        results (JSON): JSON Object (targetID, redshift) containing the targetIDs and associated 
                  redshifts for targets found in provided tileID.     
    """
    body = request.get_json()
    tileID = body['tileID']
    
    if (tileID < 1):
        raise ValueError(f'Tile ID {tileID} is invalid')                         
  
    q = db.dbSession.query(db.Ztile.targetid, db.Ztile.z).filter(db.Ztile.tileid == tileID)
    
    if (q.first() is None):
        raise ValueError(f'Tile ID {tileID} was not found')
    
    q = filter_query(q, db.Ztile, body)
    return formatJSON(q)    


@app.route('/query/zpix/', methods=['POST'])
def getRedshiftsByHEALPix():
    """ 
    @Params: 
        healpix (INTEGER): ID of HEALpix to search for redshifts
    
    @Returns:
        results (JSON): JSON Object (targetID, redshift) containing the targetIDs and associated 
                  redshifts for targets found in provided HealPIX.   
    """
    body = request.get_json()
    healpix = body['healpix']
    
    if (healpix < 1): # Set healpix bounds
        raise ValueError(f'HEALPix {healpix} is invalid')
    
    q = db.dbSession.query(db.Zpix.targetid, db.Zpix.z).filter(db.Zpix.healpix == healpix)
    
    if (q.first() is None):
        raise ValueError(f'HEALPix ID {healpix} was not found')
    
    q = filter_query(q, db.Zpix, body)
    return formatJSON(q)   


@app.route('/query/loc/', methods=['POST'])
def getRedshiftsByRADEC():
    """ 
    @Params: 
        ra (DOUBLE_PRECISION): Right Ascension of the center of cone to search for targets in degrees
        dec (DOUBLE_PRECISION): Declination of the center of cone to search for targets in degrees
        radius (DOUBLE_PRECISION): Radius of cone to search of targets in degrees
    
    @Returns:
        results (JSON): JSON Object (targetID, ra, dec, redshift) for targets found
                        in cone search of the provided ra, dec, radius
    """
    body = request.get_json()
    ra = body['ra']
    dec = body['dec']
    radius = body.get('radius', 0.01)
    if (ra > 360 or ra < 0):
        raise ValueError(f'Invalid Right Ascension {ra}')
    elif (dec > 90 or dec < -90):
        raise ValueError(f'Invalid Declination {dec}')
    elif (radius < 0):
        raise ValueError(f'Invalid Radius {radius}')
    
    q = db.dbSession.query(db.Photometry.targetid, db.Photometry.ra, db.Photometry.dec, db.Zpix.z)
    q = q.join(db.Zpix).filter(func.q3c_radial_query(db.Photometry.ra, db.Photometry.dec, ra, dec, radius))
    
    if (q.first() is None):
        raise ValueError(f'No objects found at RA {ra} and DEC {dec} within radius {radius}')
        
    q = filter_query(q, db.Zpix, body) 
    return formatJSON(q)
