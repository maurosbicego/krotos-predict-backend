from flask import Flask
from muselsl import list_muses
from flask_cors import CORS
from sqlalchemy import *
import krotos
from flask import request
from datetime import datetime
import pandas as pd
from pylsl import resolve_byprop
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.sql import and_, or_, not_
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)



engine = create_engine('sqlite:///db.sql', pool_recycle=3600)
conn = engine.connect()
metadata = MetaData(engine)
songs = Table("songs", metadata,
      Column('Id', Integer, primary_key=True, nullable=False), Column('trackid', String), Column('userid', Integer, ForeignKey('users.Id'), index=True))
users = Table("users", metadata,
      Column('Id', Integer, primary_key=True, nullable=False), Column('name', String))
samples = Table("samples", metadata,
        Column('Id', Integer, primary_key=True, nullable=False),
        Column('songid', Integer, ForeignKey('songs.Id'), index=True),
        Column('userid', Integer, ForeignKey('users.Id'), index=True),
        Column('created', DateTime, default=func.current_timestamp()),
        Column('testdata', Boolean, default=False),
        Column('0', Float),
        Column('1', Float),
        Column('2', Float),
        Column('3', Float),
        Column('4', Float),
        Column('5', Float),
        Column('6', Float),
        Column('7', Float),
        Column('8', Float),
        Column('9', Float),
        Column('10', Float),
        Column('11', Float),
        Column('12', Float),
        Column('13', Float),
        Column('14', Float),
        Column('15', Float),
        Column('16', Float),
        Column('17', Float),
        Column('18', Float),
        Column('19', Float),
        Column('20', Float),
        Column('21', Float),
        Column('22', Float),
        Column('23', Float),
        Column('24', Float),
        Column('25', Float),
        Column('26', Float),
        Column('27', Float),
        Column('28', Float),
        Column('29', Float),
        Column('30', Float),
        Column('31', Float)


      )
if not engine.dialect.has_table(engine, "songs"):
    songs.create(engine)

if not engine.dialect.has_table(engine, "samples"):
    samples.create(engine)

if not engine.dialect.has_table(engine, "users"):
    users.create(engine)

selectsamples12 = [samples.c.songid, samples.c['0'], samples.c['1'], samples.c['2'], samples.c['3'], samples.c['4'], samples.c['5'], samples.c['6'], samples.c['7'], samples.c['8'], samples.c['9'], samples.c['10'], samples.c['11']]
selectsamples32 = [samples.c.songid, samples.c['0'], samples.c['1'], samples.c['2'], samples.c['3'], samples.c['4'], samples.c['5'], samples.c['6'], samples.c['7'], samples.c['8'], samples.c['9'], samples.c['10'], samples.c['11'], samples.c['12'], samples.c['13'], samples.c['14'], samples.c['15'], samples.c['16'], samples.c['17'], samples.c['18'], samples.c['19'], samples.c['20'], samples.c['21'], samples.c['22'], samples.c['23'], samples.c['24'], samples.c['25'], samples.c['26'], samples.c['27'], samples.c['28'], samples.c['29'], samples.c['30'], samples.c['31']]

@app.route('/')
def index():
    return "Krotos"

@app.route('/listmuses')
def listmuses():
    return { 'muses': list_muses() }


@app.route('/record', methods=["POST"])
def record():
    import numpy as np
    import pandas as pd
    conn = engine.connect()
    trackid = request.get_json()['trackid']
    test = request.get_json()['test']
    uid = request.get_json()['user']
    reclen = 15
    if 'reclen' in request.get_json():
        reclen = request.get_json()['reclen']
    s = select([songs]).where((songs.c.trackid == trackid) & (songs.c.userid == uid))
    id = conn.execute(s).fetchone()
    if id:
        id = id['Id']
    else:
        res = conn.execute(songs.insert().values(trackid = trackid, userid = uid))
        id = res.inserted_primary_key[0]

    df = krotos.getFeatures(duration=int(reclen), songid=id)
    print(df)
    df['testdata'] = test
    df['userid'] = uid
    df['created'] = datetime.now()
    df.to_sql(con = conn, name = 'samples', if_exists = 'append', index = False)
    return { "status": "ok"}


@app.route('/users/add', methods=["POST"])
def addUser():
    conn = engine.connect()
    name = request.get_json()['name']
    res = conn.execute(users.insert().values(name = name))
    id = res.inserted_primary_key[0]

    s = select([users])
    usrlist = conn.execute(s)
    return { "status": "ok", "inserted": id, "users": [dict(row) for row in usrlist]}

@app.route('/users')
def getUsers():
    conn = engine.connect()
    resp = conn.execute(select([users]))
    return {'data': [dict(row) for row in resp] }


@app.route('/songs', methods=["POST"])
def getRecordedSongs():
    conn = engine.connect()
    uid = request.get_json()['user']
    resp = conn.execute(select([songs]).where(songs.c.userid == uid))
    songlist = []
    for row in resp:
        songlist.append(row)
    return {'data': [dict(row) for row in songlist] }


@app.route('/train', methods=["POST"])
def train():
    uid = request.get_json()['user']
    trainsongs = request.get_json()['songs']
    trainsongs.sort()

    modelfile = krotos.modelfname(trainsongs,uid)

    conn = engine.connect()
    df = pd.read_sql(select(selectsamples12).where(samples.c.testdata == False), conn)
    testdf = pd.read_sql(select(selectsamples12).where(samples.c.testdata == True), conn)
    df = df.loc[df['songid'].isin(trainsongs)]
    testdf = testdf.loc[testdf['songid'].isin(trainsongs)]
    X, Y = df.values[:, 1:], df.values[:, 0]
    x, y = testdf.values[:, 1:], testdf.values[:, 0]
    X = StandardScaler().fit_transform(X)
    x = StandardScaler().fit_transform(x)
    X = X.astype('float32')
    x = x.astype('float32')
    Y = LabelEncoder().fit_transform(Y)
    y = LabelEncoder().fit_transform(y)
    Y = krotos.padlabels(Y)
    y = krotos.padlabels(y)


    acc = krotos.train(X,Y,x,y,modelfile)
    if acc:
        return { "status": "ok", "modelfile": modelfile, "accuracy": acc }
    else:
        return { "status": "not ok" }, 500

@app.route('/comparealgos')
def comparealgos():
    from sklearn import metrics
    from sklearn.preprocessing import MinMaxScaler
    import autokeras as ak

    trainsongs = [1,2]
    uid = 1
    modelfile = krotos.modelfname(trainsongs,uid)
    conn = engine.connect()
    df = pd.read_sql(select(selectsamples12).where(samples.c.testdata == False), conn)
    testdf = pd.read_sql(select(selectsamples12).where(samples.c.testdata == True), conn)
    df = df.loc[df['songid'].isin(trainsongs)]
    testdf = testdf.loc[testdf['songid'].isin(trainsongs)]
    X, Y = df.values[:, 1:], df.values[:, 0]
    x, y = testdf.values[:, 1:], testdf.values[:, 0]
    X = StandardScaler().fit_transform(X)
    x = StandardScaler().fit_transform(x)
    X = X.astype('float32')
    x = x.astype('float32')
    Y = LabelEncoder().fit_transform(Y)
    y = LabelEncoder().fit_transform(y)

    Y = krotos.padlabels(Y)
    y = krotos.padlabels(y)
    krotos.train(X,Y,x,y,modelfile)




    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X, Y)
    pred = np.array(clf.predict(x))
    acc = metrics.accuracy_score(y, pred)*100


    return { "status": "ok"}




@app.route('/predict', methods=["POST"])
def predict():
    import autokeras as ak

    iterations = 3
    uid = request.get_json()['user']
    trainsongs = request.get_json()['songs']
    trainsongs.sort()
    reclen = 15
    if 'reclen' in request.get_json():
        reclen = request.get_json()['reclen']

    if 'iterations' in request.get_json():
        iterations = request.get_json()['iterations']

    print(np.argmax([20,30]))
    modelfile = krotos.modelfname(trainsongs,uid)
    print(modelfile)
    if os.path.isfile(modelfile):
        model = load_model(modelfile, custom_objects=ak.CUSTOM_OBJECTS)
    else:
        return { "status": "error", "reason": "train first" }, 500

    recordings = []
    for iteration in range(int(iterations)):
        recordings.append(krotos.getFeatures(duration=reclen))



    predictions = [None] * iterations
    i = 0
    n_features = 12
    for recording in recordings:
        recording = StandardScaler().fit_transform(recording)
        recording = recording.astype('float32')
        for row in np.array(recording):
            row = np.array([row[:n_features]])
            prediction = model.predict(row)
            if predictions[i] is not None:
                predictions[i] += np.array(prediction)
            else:
                predictions[i] = np.array(prediction)
        i += 1

    ret = []
    preds = []
    for prediction in predictions:
        ret.append(prediction.tolist())
        print(prediction)
        print(np.argmax(prediction))
        preds.append(trainsongs[np.argmax(prediction)])

    return { "predictions": ret, "ids": preds, "overall":  int(np.bincount(preds).argmax())}

@app.route('/stream/check')
def checkstream():
    streams = resolve_byprop('type', 'EEG', timeout=2)

    if len(streams) == 0:
        return { 'data': False }
    else:
        return { 'data': True }

if __name__ == '__main__':
    app.run(debug=True)
