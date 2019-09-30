from pymongo import MongoClient
from random import randint
#Step 1: Connect to MongoDB - Note: Change connection string as needed
client = MongoClient(port=27017)
# Create collection named "tech_jobs"
db=client.tech_jobs
#Step 2: Create sample data
titre = ['a', 'b', 'c']
company = ['d', 'e', 'f']
adress = ['1a', '2b', '3c']
salary = [1, 2, 3]
job_description = ['abab', 'blabla', 'laclac']
date = ['1/a', '2/b', '3/c']
# input the data 
for x in range(len(titre)):
    indeed_jobs = {
        'titre' : titre[x],
        'company' : company[x],
        'adress' : adress[x],
        'salary' : salary[x],
        'job_description' : job_description[x],
        'date' : date[x] 
    }
    #Step 3: Insert business object directly into MongoDB via insert_one
    result=db.indeed_jobs.insert_one(indeed_jobs)
    #Step 4: Print to the console the ObjectID of the new document
    print('Created {0} of 100 as {1}'.format(x,result.inserted_id))
#Step 5: Tell us that you are done
print('finished uploading collected data to database tech_jobs - collections indeed_jobs')