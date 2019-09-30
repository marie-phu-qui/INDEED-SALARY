from pymongo import MongoClient
from random import randint
#Step 1: Connect to MongoDB - Note: Change connection string as needed
client = MongoClient(port=27017)
# Create collection named "indeed_jobs"
db=client.indeed_jobs
#Step 2: Create sample data
titre = ['a', 'b', 'c']
company = ['d', 'e', 'f']
adress = ['1a', '2b', '3c']
salary = [1, 2, 3]
job_description = ['abab', 'blabla', 'laclac']
date = ['1/a', '2/b', '3/c']
# generate 500 random data
for x in range(1, 501):
    indeed_jobs = {
        'titre' : titre[randint(0, (len(titre)-1))],
        'company' : company[randint(0, (len(company)-1))],
        'adress' : adress[randint(0, (len(adress)-1))],
        'salary' : randint(0, 5),
        'job_description' : job_description[randint(0, (len(job_description)-1))],
        'date' : date[randint(0, (len(date)-1))] 
    }
    #Step 3: Insert business object directly into MongoDB via isnert_one
    result=db.indeed_jobs.insert_one(indeed_jobs)
    #Step 4: Print to the console the ObjectID of the new document
    print('Created {0} of 100 as {1}'.format(x,result.inserted_id))
#Step 5: Tell us that you are done
print('finished creating 100 business reviews')