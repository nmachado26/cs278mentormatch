import pandas as pd
from jinja2 import Template
import requests

#from sklearn.preprocessing import StandardScaler
#from sklearn.neighbors import NearestNeighbors
from math import dist
import numpy as np
import csv
#import matplotlib.pyplot as plt

#import csv of profs and of students
#students = pd.read_csv('Student.csv')
#teachers = pd.read_csv('Teacher.csv')
responses = pd.read_csv('responses_current.csv')
# responses = pd.read_csv('responses_test.csv')


#clean up data
def clean_up_data_students(df):
    to_remove_columns_students = [
        '#',
        'What is your Stanford email address?',
        'Enter the password for mentors.',
        'What is your name?',
        'What are your pronouns?',
        'What major or academic department are you interested in?',
        'Academic Interest #2: What major or academic department are you interested in?',
        'Academic Interest #3: What major or academic department are you interested in?',
        'More school',
        'Public sector work',
        'Private sector work',
        'Volunteer work',
        'Travel/Gap year',
        'Start-up',
        'Not sure',
        'I need some sort of structure in my life..1',
        'I would describe myself as an artistic/creative person..1',
        'I prefer a night in more than a night out..1',
        'I would describe myself as analytically oriented..1',
        'I enjoy spending time outdoors..1',
        'I would describe myself as introverted..1',
        'I stay up to date with current events (eg. by watching and/or reading the news)..1',
        'I believe everything happens for a reason..1',
        'I enjoy participating in protests and displays of social activism..1',
        'I enjoy talking about politics..1',
        'I enjoy engaging in deep and intellectual conversations with friends..1',
        'I think science can/will explain everything..1',
        'I enjoy cooking..1',
        'I believe that people are inherently good..1',
        'When going on vacation, I tend to plan everything ahead of time..1',
        'I tend to live in the moment..1',
        'I spend a lot of time reflecting..1',
        'I consider myself a "Type A" person rather than a "Type B" person..1',
        'When I hear the right song, there is nothing stopping me from dancing..1',
        'I am not afraid to sing out loud with my friends..1',
        'I enjoy dinner parties..1',
        'I am an avid traveler..1',
        'I need some sort of structure in my life..1',
        'I tend to see the glass half full..1',
        'What is your name?.1',
        'What are your pronouns?.1',
        'What is the primary academic department you are affiliated with?',
        'Academic Department 2: What other academic departments are you affiliated with or interested in? [Optional]',
        'Academic Department 3: What other academic departments are you affiliated with or interested in? [Optional]',
        'More school.1',
        'Public sector work.1',
        'Private sector work.1',
        'Volunteer work.1',
        'Travel/Gap year.1',
        'Start-up.1',
        'Not sure.1',
        'Your Bio',
        'Upload a photo',
        'Upload a photo.1',
        'Upload a photo.2',
        'Upload a photo.3',
        'Upload a photo.4',
        'Upload a photo.5',
        'Upload a photo.6',
        'Upload a photo.7',
        'Upload a photo.8',
        'Upload a photo.9',
        'Start Date (UTC)',
        'Submit Date (UTC)',
        'Network ID'
    ]
    return df.drop(columns=to_remove_columns_students)

#clean up data
def clean_up_data_teachers(df):
    to_remove_columns_teachers = [
        '#',
        'What is your Stanford email address?',
        'Enter the password for mentors.',
        'What is your name?',
        'What are your pronouns?',
        'What major or academic department are you interested in?',
        'Academic Interest #2: What major or academic department are you interested in?',
        'Academic Interest #3: What major or academic department are you interested in?',
        'More school',
        'Public sector work',
        'Private sector work',
        'Volunteer work',
        'Travel/Gap year',
        'Start-up',
        'Not sure',
        'I need some sort of structure in my life.',
        'I would describe myself as an artistic/creative person.',
        'I prefer a night in more than a night out.',
        'I would describe myself as analytically oriented.',
        'I enjoy spending time outdoors.',
        'I would describe myself as introverted.',
        'I stay up to date with current events (eg. by watching and/or reading the news).',
        'I believe everything happens for a reason.',
        'I enjoy participating in protests and displays of social activism.',
        'I enjoy talking about politics.',
        'I enjoy engaging in deep and intellectual conversations with friends.',
        'I think science can/will explain everything.',
        'I enjoy cooking.',
        'I believe that people are inherently good.',
        'When going on vacation, I tend to plan everything ahead of time.',
        'I tend to live in the moment.',
        'I spend a lot of time reflecting.',
        'I consider myself a "Type A" person rather than a "Type B" person.',
        'When I hear the right song, there is nothing stopping me from dancing.',
        'I am not afraid to sing out loud with my friends.',
        'I enjoy dinner parties.',
        'I am an avid traveler.',
        'I need some sort of structure in my life.',
        'I tend to see the glass half full.',
        'What is your name?.1',
        'What are your pronouns?.1',
        'What is the primary academic department you are affiliated with?',
        'Academic Department 2: What other academic departments are you affiliated with or interested in? [Optional]',
        'Academic Department 3: What other academic departments are you affiliated with or interested in? [Optional]',
        'More school.1',
        'Public sector work.1',
        'Private sector work.1',
        'Volunteer work.1',
        'Travel/Gap year.1',
        'Start-up.1',
        'Not sure.1',
        'Your Bio',
        'Upload a photo',
        'Upload a photo.1',
        'Upload a photo.2',
        'Upload a photo.3',
        'Upload a photo.4',
        'Upload a photo.5',
        'Upload a photo.6',
        'Upload a photo.7',
        'Upload a photo.8',
        'Upload a photo.9',
        'Start Date (UTC)',
        'Submit Date (UTC)',
        'Network ID'

    ]
    return df.drop(columns=to_remove_columns_teachers)

students = responses[responses['Are you a student or a mentor?'] == 'Student']
teachers = responses[responses['Are you a student or a mentor?'] == 'Mentor']
students = students.drop('Are you a student or a mentor?', 1)
teachers = teachers.drop('Are you a student or a mentor?', 1)
full_student = students
full_teacher = teachers
students = clean_up_data_students(students)
teachers = clean_up_data_teachers(teachers)

print('students \n')
print(students)
print('\nteachers \n')
print(teachers)


shortest_distances = [] #each index is a student, each value is the (teacher match i, dist value)

def get_distances(students, teachers):

    #iterate through each student, find a teacher match
    for i, s_row in students.iterrows():
        min_dist = None
        min_dist_index = 0
        rankings = {}
        for j, t_row in teachers.iterrows():
            vals_student = s_row.to_numpy()
            vals_teacher = t_row.to_numpy()
            #print("\nstudent: ", vals_student)
            #print("teacher: ", vals_teacher)
            dist =  vals_student - vals_teacher
            #print("dist: ", dist)
            dist_final = np.linalg.norm(dist)
            rankings[j] = dist_final
            # #print("dist final :", dist_final)
            # if min_dist is None or dist_final < min_dist:
            #     min_dist = dist_final
            #     min_dist_index = j
        # rankings.sort(key = lambda x: x[0])
        shortest_distances.append((i, rankings))
get_distances(students, teachers)
students_per_teacher = 17
new_shortest_distance = []
print(shortest_distances)
for j, t_row in teachers.iterrows():
    matches = []
    shortest_distances.sort(key = lambda x: x[1][j])
    for i in range(students_per_teacher):
        if len(shortest_distances):
            student = shortest_distances.pop(0)
            matches.append(student)
            new_shortest_distance.append((j, student[0], student[1][j]))

new_shortest_distance.sort(key = lambda x: x[1])
new_list = []
for j, i, score in new_shortest_distance:
    new_list.append((j, score))
shortest_distances = new_list
print('\nshortest distances \n')
print(shortest_distances)

def format_email(studentName, studentEmail, mentorName, mentorEmail, pronouns, mentorDept, bio, image):
    rendered = Template(open('email_template.html').read()).render(
        studentName=studentName,
        studentEmail=studentEmail,
        mentorName=mentorName,
        mentorEmail=mentorEmail,
        pronouns=pronouns,
        mentorDept=mentorDept,
        bio=bio,
        image=image
    )
    return rendered

def send_to_smtp(receiver, message):
    url = 'https://api.sendinblue.com/v3/smtp/email'
    headers = {"accept": 'application/json',
            'api-key': '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',
            'content-type': 'application/json'}
    data = {
            "sender":{
               "name": "Team MentorMatch",
               "email":"hello@mentormatch.su.domains"
            },
            "to": receiver,
            "subject": "New Mentor!",
            "htmlContent": message
        }
    x = requests.post(url, json = data, headers = headers)
    print(x.text)


def email_student_match(studentName, studentEmail, mentorName, mentorEmail, pronouns, mentorDept, bio, image):
    message = format_email(studentName, studentEmail, mentorName, mentorEmail, pronouns, mentorDept, bio, image)
    send_to_smtp([{"name": studentName, "email": studentEmail}], message)

counter = {}
print("Matching...")

for i in range(len(shortest_distances)):
    student = full_student.iloc[i]
    mentor = responses.iloc[shortest_distances[i][0]]
    mentor_name = mentor['What is your name?.1']
    if mentor_name not in counter:
        counter[mentor_name] = 1
    else:
        counter[mentor_name] += 1
    print(student['What is your name?'], "---", mentor['What is your name?.1'])
    # Uncomment to have email actually send
    # email_student_match(
    #     student['What is your name?'],
    #     student['What is your Stanford email address?'],
    #     mentor['What is your name?.1'],
    #     mentor['What is your Stanford email address?'],
    #     mentor['What are your pronouns?.1'],
    #     mentor['What is the primary academic department you are affiliated with?'],
    #     mentor['Your Bio'],
    #     mentor['Upload a photo']
    # )
print(counter)