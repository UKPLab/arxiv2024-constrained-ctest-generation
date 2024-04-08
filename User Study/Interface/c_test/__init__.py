import os
import sys
import csv
import json
import random
import requests
import hashlib
from datetime import datetime, timedelta, date
from json.decoder import JSONDecodeError

from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, Blueprint, jsonify
from jinja2 import TemplateNotFound

from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

import logging
logging.basicConfig(stream=sys.stderr)

app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'c-test.db'),
    USERNAME='admin',
    PASSWORD='admin',
    SECRET_KEY = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
))

engine = create_engine('mysql+mysqlconnector://admin:admin@localhost/c-test', 
        encoding='utf-8', 
        pool_recycle=3600, 
        connect_args={'auth_plugin': 'mysql_native_password'})
Base = declarative_base(engine)

##################################################################
#		DATABASE TABLES
##################################################################

    
class User(Base):
    __tablename__ = 'user'
    __table_args__ = {'autoload':True}

class CTest(Base):
    __tablename__ = 'ctest'
    __table_args__ = {'autoload':True}
    
class Questionnaire(Base):
    __tablename__ = 'questionnaire'
    __table_args__ = {'autoload':True}

class Strategy(Base):
    __tablename__ = 'strategy'
    __table_args__ = {'autoload':True}

class CTest_User_Mapping(Base):
    __tablename__ = 'ctest_user_mapping'
    __table_args__ = {'autoload':True}

    
##################################################################
#		DATABASE FUNCTIONS
##################################################################


# Create session with all tables 
def create_session():
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
    
def get_strategy(strategy_name):
    session = create_session()
    result = session.query(Strategy).filter_by(name=strategy_name).first().annotation_path
    session.close()
    return result
    
def check_user_credentials(username):
    session = create_session()
    try:
        user_id = session.query(User).filter_by(username=username).first().id
        session.commit()
        session.close()
        return True
        
    except AttributeError:
        session.commit()
        session.close()
        return False

def add_user(username):
    session = create_session()
    try:
        user_id = session.query(User).filter_by(username=username).first().id
        session.commit()
        session.close()
        return False
        
    except AttributeError:
        strategies = session.query(Strategy).all()
        strategy_counter = {strat.id:0 for strat in strategies}
        all_users = session.query(User).all()
        
        for user in all_users:
            strategy_counter[user.strategy] += 1
            
        inversed_strategies = {v:k for k,v in strategy_counter.items()}
        strategy_to_set = sorted(inversed_strategies.items())[0][1]
        # Set user credentials and sampling strategy
        session.add(User(username=username.encode('utf-8'), strategy=strategy_to_set, finished=0, questionnaire=0, consent=0))
        session.commit()
        user_id = session.query(User).filter_by(username=username).first().id
        
    session.commit()
    session.close()
    return True
    
def remove_user(user_id):
    session = create_session()
    user =  session.query(User).filter_by(id=user_id).first()
    session.delete(user)
    session.commit()
    session.close()

# NOTE: Only use, when logged in!
def get_user_id(username):
    session = create_session()
    user_id = session.query(User).filter_by(username=username).first().id
    session.close()
    return user_id
    
def check_user_consent(user_id):
    session = create_session()
    result = session.query(User).filter_by(id=user_id).first().consent
    session.close()
    if result == 0:
        return False
        
    else:
        return True
        
def set_user_consent(user_id):
    session = create_session()
    user = session.query(User).filter_by(id=user_id).first()
    user.consent = 1
    session.commit()
    session.close()
        

def get_strategy_name(user_id):
    session = create_session()
    strategy_id = session.query(User).filter_by(id=user_id).first().strategy
    strategy_name = session.query(Strategy).filter_by(id=strategy_id).first().name
    session.close()
    return strategy_name

def set_user_is_done(user_id):
    session = create_session()
    result = session.query(User).filter_by(id=user_id).first()
    result.finished = 1
    session.commit()
    session.close()


def check_user_is_done(user_id):
    session = create_session()
    result = session.query(User).filter_by(id=user_id).first().finished
    session.close()
    if result == 0:
        return False
        
    else:
        return True
        
        
def check_user_is_done_questionnaire(user_id):
    session = create_session()
    result = session.query(User).filter_by(id=user_id).first().questionnaire
    session.close()
    if result == 0:
        return False
    else:
        return True
            
        
def get_finished_ctests(user_id):
    session = create_session()
    ctests_taken = session.query(CTest_User_Mapping).filter_by(user_id=user_id).all()
    result = [ctest.ctest_id for ctest in ctests_taken]
    session.close()
    return result

    
def store_ctest_feedback(ctest_id, user_id, responses, time, perceived_difficulty):
    session = create_session()
    session.add(CTest_User_Mapping(
                           ctest_id=ctest_id, 
                           user_id=user_id,
                           responses=responses, 
                           time=time, 
                           perceived_difficulty=perceived_difficulty
                        ))
    session.commit()
    session.close()
    
    
# Get the number of ctests a user has already done
def get_ctest(ctest_id):
    result = {"id":ctest_id,
              "ctest":"",
              "gaps":"",
              "plaintext":"",
              "tokens":""  }
    session = create_session()
    subres = session.query(CTest).filter_by(id=ctest_id).first()
    result["ctest"] = subres.ctest
    result["gaps"] = subres.gaps
    result["plaintext"] = subres.plaintext
    result["tokens"] = subres.tokens
    session.close()
    return result

# Function for fetching a new ctest the user has not done (or at least done the least number of times)
def get_next_ctest(user_id):
    session = create_session()
    # Fetch all ctests the user already has done:
    user_ctests = session.query(CTest_User_Mapping).filter_by(user_id=user_id).all()
    last_index = len(user_ctests)
    # Get the sampling strategy:
    strategy_id = session.query(User).filter_by(id=user_id).first().strategy
    strategy_query = session.query(Strategy).filter_by(id=strategy_id).first()
    strategy_rotation = {0:strategy_query.test_1,1:strategy_query.test_2,2:strategy_query.test_3,3:strategy_query.test_4}
    rotation_strategy = {v:k for k,v in strategy_rotation.items()}
    # first exercise
    if last_index < 1:
         return get_ctest(strategy_rotation[0])
    # last exercise
    if last_index >= 4:
        return False
    finished = get_finished_ctest_ids(user_id)  
    # Check if the user has already done the ctest!
    if strategy_rotation[last_index] in finished:
        # Fetch ctest that has not been done yet
        ftmp = [key for key in rotation_strategy.keys() if key not in finished]
        return get_ctest(ftmp[0])
    return get_ctest(strategy_rotation[last_index])

# Get the number of ctests a user has already done
def get_num_finished_ctests(user_id):
    result = 0
    session = create_session()
    result = len(session.query(CTest_User_Mapping).filter_by(user_id=user_id).all())
    session.close()
    return result

# Get the number of ctests a user has already done
def get_finished_ctest_ids(user_id):
    result = []
    session = create_session()
    ctests_done = session.query(CTest_User_Mapping).filter_by(user_id=user_id).all()
    result = [x.ctest_id for x in ctests_done]
    session.close()
    return result

def store_questionnaire_results(user_id, data):
    session = create_session()
    session.add(Questionnaire(
                       user_id=user_id, 
                       cefr=data['cefr'], 
                       learning_years=data['years'], 
                       frequency=data['frequency'],
                       native_language=data['native-tongue'],
                       other_languages=data['other-languages'],
                       other_languages_responses=data['other-languages-responses']))
    user = session.query(User).filter_by(id=user_id).first()
    user.questionnaire = 1
    session.commit()
    session.close()

# Get all ctest error rates for all languages and return them as a dict
def get_statistics(user_id):
    result = dict()
    session = create_session()
    ctests_taken = session.query(CTest_User_Mapping).filter_by(user_id=user_id).all()
    for ctest in ctests_taken:
        time_taken = ctest.time.split('.')[0][3:].replace(':','m ')+'s'
        result[ctest.id] = {'ctest_id':ctest.ctest_id, 'score':ctest.score, 'ctest_taken_date':ctest.ctest_taken_date, 'user_feedback':ctest.perceived_difficulty, 'time_taken':time_taken}
    session.close()
    return result

def store_ctest_results(user_id, results, ctest_taken_date):
    session = create_session()
    ctest_id = results['ctest_id']
    answers = [str(answer['id']) + '-' + answer['answer'] for answer in results['results']]
    errors = [str(error['correct']) for error in results['results']]
    total = str(len(errors))
    correct = str(sum([int(err) for err in errors]))
    session.add(CTest_User_Mapping(ctest_id=ctest_id, user_id=user_id, responses=' '.join(answers), errors=' '.join(errors), score=correct+'/'+total, time=results['time_taken'], ctest_taken_date=ctest_taken_date))
    session.commit()
    session.close()
    
# User feedback is set to NULL by default
def add_ctest_feedback(c_test_id, user_id, feedback, point_assessment):
    session = create_session()
    # Get the last taken ctest
    user_c_tests = session.query(CTest_User_Mapping).filter_by(user_id=user_id, ctest_id=c_test_id).all()
    test_id_date = dict()
    # Convert to dict
    for key in user_c_tests:
        test_id_date[key.id] = key.ctest_taken_date
    # Get the most recent c test taken by the user
    most_recent_id = sorted(test_id_date.items(), key=lambda p: p[1], reverse=True)[0]
    correct_instance = session.query(CTest_User_Mapping).filter_by(user_id=user_id, ctest_id=c_test_id, id=most_recent_id[0]).first()
    # Get the feedback and set it
    correct_instance.perceived_difficulty = feedback
    correct_instance.self_estimation = point_assessment
    session.commit()
    session.close()

def get_ctest_result(ctest_taken_id):
    session = create_session()
    # Get errors
    ctest_taken = session.query(CTest_User_Mapping).filter_by(id=ctest_taken_id).first()
    total_num_correct, total_gap_number = ctest_taken.score.split('/')
    # Get ctest_data
    ctest_data_db = session.query(CTest).filter_by(id=ctest_taken.ctest_id).first()
    ctest_data = []
    counter = 0
    for token in ctest_data_db.ctest.split():
        if token.endswith("#GAP#"):
            ctest_data.append({'token':token, 'number':counter})
            counter += 1
        else:
            ctest_data.append({'token':token, 'number':-1})
    # Set correct answers and user answers
    errors = []
    for gap_id,answer,true in zip(ctest_taken.responses.split(), ctest_taken.errors.split(),ctest_data_db.gaps.split()):
        user_answer = gap_id.split('-')[-1].strip()
        errors.append({'answer':user_answer, 'true':true, 'correct':int(answer)})
    result = {'result':errors, 'ctest_data':ctest_data, 'correct':total_num_correct, 'num_gaps':total_gap_number, 'time_taken':ctest_taken.time.split('.')[0]}
    session.close()
    return result

def get_ctest_id(ctest_taken_id):
    session = create_session()
    ctest_taken = session.query(CTest_User_Mapping).filter_by(id=ctest_taken_id).first()
    result = ctest_taken.ctest_id 
    session.close()
    return result

def get_text_stats(ctest_taken_id):
    print(ctest_taken_id)
    session = create_session()
    # Get errors
    ctest_taken = session.query(CTest).filter_by(id=ctest_taken_id).first()
    print(ctest_taken)
    method, text_id = ctest_taken.ctest_type.split(' ')
    if "pride_and_prejudice" in text_id:
        return {"title":"Pride and Prejudice" , "author":"Jane Austen" , "source":"https://www.gutenberg.org/ebooks/1342" }
    elif "emma" in text_id:
        return {"title":"Emma" , "author":"Jane Austen" , "source":"https://www.gutenberg.org/ebooks/158" }
    elif "beyond_good_and_evil" in text_id:
        return {"title":"Beyond Good and Evil" , "author":"Friedrich Nietzsche" , "source":"https://www.gutenberg.org/ebooks/4363" }
    else: #  "crime_and_punishment" in text_id
        return {"title":"Crime and Punishment" , "author":"Fyodor Dostoyevsky" , "source":"https://www.gutenberg.org/ebooks/2554" }

##################################################################
#		Internal Helper Functions
##################################################################

def get_token_indexes(tokens, text):
    result = []
    string_index = 0
    working_text = text
    for i, token in enumerate(tokens):
        string_index += working_text.index(token)
        result.append((token, string_index, string_index+len(token)))
        string_index += len(token)
        working_text = text[string_index:]
        if len(working_text) == 0:
            # If there is no more text, return.
            break

    return result
    
def generate_ctest_html(c_test_ready):
    token_indexes = get_token_indexes(c_test_ready['tokens'].split(),' '.join(c_test_ready['plaintext'].split()))
    c_test_index_data = [{'number':i,'token':token} for i, token in enumerate(zip(c_test_ready['ctest'].split(),token_indexes))]
    gaps = c_test_ready['gaps'].split()
    # Patch together the html for the template with original spacing
    c_test_text = ''
    end_index = 0
    gap_index = 0
    for tokendict in c_test_index_data:
        if tokendict['token'][1][1] > end_index:
            c_test_text += ' ' * (tokendict['token'][1][2]-end_index)
        if tokendict['token'][0].endswith('#GAP#'):
            number = str(tokendict['number'])
            c_test_text += '<nobr>'+tokendict['token'][0][:-5]+'<input class="ctest" type="text" onchange="add_token_timestamp(event)"  autocomplete="off" name="'+number+'" id="'+number+'" size="'+str(len(gaps[gap_index]))+'" maxlength="'+str(len(gaps[gap_index]))+'" gap-size="'+ str(len(gaps[gap_index])) +'"></nobr>'
            gap_index += 1
        else:
            c_test_text += tokendict['token'][1][0]
        end_index = tokendict['token'][1][2]
    c_test_data = {'id':c_test_ready['id'], 'ctest_text':c_test_text}
    return c_test_data
    
# Patch together html with original spacing:
def generate_ctest_result_html(c_test_true, errors, drag=True):
    token_indexes = get_token_indexes(c_test_true['tokens'].split(),' '.join(c_test_true['plaintext'].split()))
    c_test_index_data = [{'number':i,'token':token} for i,token in enumerate(zip(c_test_true['ctest'].split(),token_indexes))]
    # Patch together the html for the template with original spacing
    c_test_result_text = ''
    end_index = 0
    gap_index = 0
    for tokendict in c_test_index_data:
        if tokendict['token'][1][1] > end_index:
            c_test_result_text += ' ' * (tokendict['token'][1][2]-end_index)
        if tokendict['token'][0].endswith('#GAP#'):
            # Define gaps with individual spans
            gap_span = '<gap class="gap" id="'+str(gap_index)+'">'
            if drag:
                gap_span = gap_span.replace('">','" classified="false"><input type="hidden" value="none" name="'+str(gap_index)+'">') 
            current_tok = errors[gap_index]['answer']
            if current_tok.strip() == '':
                current_tok += '_' * len(errors[gap_index]['true'])
            c_test_result_text += '<nobr>'+gap_span+tokendict['token'][0][:-5]
            # We have a gap, get the result id and patch together HTML 
            if errors[gap_index]['correct'] == 0:
                c_test_result_text += '<span class="wrong" id="wrong-'+str(gap_index)+'">'+current_tok+'</span><span id="click-'+str(gap_index)+'" class="correction" > ('+tokendict['token'][0][:-5]+errors[gap_index]['true']+')</span>'
            else:
                c_test_result_text += '<span class="correct" id="correct-'+str(gap_index)+'">'+errors[gap_index]['true']+'</span>'
            c_test_result_text += '</gap></nobr>' #<input id="click-'+str(gap_index)+'" type="button" onClick="add_mobile(event)" style="height:20px;width:20px" >
            gap_index += 1
        else:
            c_test_result_text += tokendict['token'][0]
        end_index = tokendict['token'][1][2]
    return c_test_result_text

    
##################################################################
#		WEBEND
##################################################################

# Starting page for the exercise generator
@app.route('/')
@app.route('/index', methods = ['POST','GET'])
def index(name=None):
    return render_template('study_index.html', name=name, data={})

@app.route('/study_task_description')
def task_description(name=None):
    return render_template('study_ctest_description.html', name=name, data={})

@app.route('/study_task_description_test')
def task_description_test(name=None):
    return render_template('study_ctest_description_test.html', name=name, data={})


@app.route('/study_cefr_description')
def cefr_description(name=None):
    return render_template('study_cefr_description.html', name=name, data={})

@app.route('/informed_consent')
def informed_consent(name=None):
    return render_template('study_informed_consent_1.html', name=name, data={})
    
@app.route('/next')
def informed_consent_2(name=None):
    return render_template('study_informed_consent_2.html', name=name, data={})

@app.route('/agree')
def give_consent(name=None):
    set_user_consent(session['user_id'])
    return task_description()
    
@app.route('/disagree')
def reject_consent(name=None):
    flash('No worries; we have deleted your participation key. If you reconsider your participation please register anew.')
    remove_user(session['user_id'])
    return render_template('study_thank_you.html', name=name, data={})

@app.route('/questionnaire')
def questionnaire(name=None):
    return render_template('study_questionnaire.html', name=name, data={})

@app.route('/thank_you')
def thank_you(name=None):
    if session['logged_in']:
        return render_template('study_thank_you.html', name=name, data={})
        
    return index()


@app.route('/register')
def registrate(name=None):
    return render_template('study_registrate.html', name=name, data={})

# Add a new user
@app.route('/create_user', methods=['POST'])
def create_user(name=None):
    userkey = request.form['username']
    username = hashlib.sha256(userkey.encode('utf-8')).hexdigest()
    if not add_user(username):
        flash('Username {} is already taken. Please select a different one!'.format(userkey))
        return registrate()
    else:
        flash('Registrated user {}. Thank you for registering!'.format(userkey))
        return study()

# Login 
@app.route('/login', methods=['POST','GET'])
def login(name=None):
    if request.method == 'POST':
        userkey = request.form['username']
        username = hashlib.sha256(userkey.encode('utf-8')).hexdigest()
        if check_user_credentials(username):
            session['logged_in'] = True
            session['user_id'] = get_user_id(username)
            return study()
        else:
            flash('Non existing user!')
            
        return index()
        
    else:
        return render_template('study_login.html')

# Logout
@app.route('/logout', methods=['POST','GET'])
def logout(name=None):
    session['logged_in']=False
    return index()

# Get the next ctest
@app.route('/study', methods = ['GET','POST'])
def study(name=None):
    if not session.get('logged_in'):
        return render_template('study_login.html')
        
    if not check_user_consent(session['user_id']):
        return informed_consent()
        
    if not check_user_is_done_questionnaire(session['user_id']):
        return questionnaire()
    
    c_test_ready = get_next_ctest(session['user_id'])
    # Check if the user is done with all ctests:
    if not c_test_ready:
        # Set flag for this user
        set_user_is_done(session['user_id'])
        return thank_you()
    data = generate_ctest_html(c_test_ready)
    # Extract the information out of the input file name:
    data['date']=datetime.today()

    return render_template('study_ctest.html',name=name,data=data)

        
@app.route('/finish_questionnaire', methods = ['POST'])
def finish_questionnaire(name=None):
    data = {
        'cefr':request.form['cefr'],
        'years':request.form['years'],
        'frequency':request.form['frequency'],
        'native-tongue':request.form['native-tongue'],
        'other-languages':request.form['other-languages'],
        'other-languages-responses':request.form['other-languages-responses']
    }
        
    store_questionnaire_results(session['user_id'], data)
    return study()

# Get the next ctest
@app.route('/ctest', methods = ['GET','POST'])
def ctest(name=None,show_questionnaire=True):
    if not session.get('logged_in'):
        return render_template('study_login.html')
    if check_user_is_done(session['user_id']):
        return thank_you()
    # Show questionnarie for the first login:
    if get_num_finished_ctests(session['user_id']) == 0 and show_questionnaire:
        return render_template('study_questionnaire.html', name=name, data={})
    # Check for new ctests as long as we haven't looked at all texts:
    c_test_ready = get_next_ctest(session['user_id'])
    # Check if the user is done with all ctests:
    if not c_test_ready:
        # Set flag for this user
        set_user_is_done(session['user_id'])
        return thank_you()
    c_test_data = generate_ctest_html(c_test_ready)
    # Extract the information out of the input file name:
    c_test_data['date']=datetime.today()
    return render_template('study_ctest.html',name=name, data=c_test_data)
    
# Evaluate c tests 
@app.route('/score_c_test', methods = ['POST'])
def score_c_test(name=None):
    if not session.get('logged_in'):
        return render_template('study_login.html')
    # Get the results
    c_test_id = int(request.form['c_test_id'])
    answers = sorted([(int(gap_id), gap_str) for gap_id, gap_str in dict(request.form).items() if gap_id.isnumeric()])
    cleaned_answers = sorted([(i, element[1]) for i,element in enumerate(answers)])
    # Get correct answers:
    c_test_true = get_ctest(c_test_id)
    # Evaluate C Test:
    gaps = c_test_true['gaps'].split()
    assert(len(gaps)==len(cleaned_answers))
    total_gap_number = len(gaps)
    total_num_correct = 0
    errors = []
    for (gap_id,answer),true in zip(cleaned_answers,gaps):
        res = {'id':gap_id, 'answer':answer, 'true':true, 'correct':0}
        if answer.strip() == true.strip():
            total_num_correct+=1
            res['correct']=1
        errors.append(res)
    # Add new ctest - result entry
    started = request.form['startingdate']
    finish = datetime.today()
    time_taken = finish - datetime.strptime(started, '%Y-%m-%d %H:%M:%S.%f')
    store_ctest_results(session['user_id'], {'results':errors, 'ctest_id':c_test_id, 'time_taken':time_taken}, finish)
    return render_template('study_ctest_result.html',name=name, data={ 'correct':total_num_correct, 'num_gaps':total_gap_number, 'ctest_id':c_test_id, 'time_taken':str(time_taken).split('.')[0]}) 

# Show statistics of the current user
@app.route('/stats', methods = ['GET','POST'])
def stats(name=None):
    if not session.get('logged_in'):
        return render_template('exgen_login.html')
    user_id = session['user_id']
    statistics = get_statistics(user_id)
    languages = dict()
    return render_template('study_statistics.html',name=name, data={'statistics':statistics})

# Show the answers for a previous c test:
@app.route('/show_single_result', methods = ['GET'])
def show_single_result(name=None):
    errors = get_ctest_result(int(request.args['ctest_taken_id']))
    c_test_id = get_ctest_id(int(request.args['ctest_taken_id']))
    c_test_true = get_ctest(c_test_id)
    textdict = get_text_stats(c_test_id)
    return render_template('study_show_old_result.html',name=name, data={'html':generate_ctest_result_html(c_test_true, errors['result'], False), 'correct':errors['correct'], 'num_gaps':errors['num_gaps'], 'time_taken':errors['time_taken'], 'source_publisher':textdict['source'], 'source_title':textdict['title'], 'source_author':textdict['author']})


@app.route('/send_feedback', methods = ['GET','POST'])
def send_feedback(name=None):
    if not session.get('logged_in'):
        return render_template('study_login.html')
    add_ctest_feedback(request.form['c_test_id'], session['user_id'], request.form['difficulty'], request.form['point_assessment'])
    return ctest(show_questionnaire=False)

    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)





