# BA_Project

### How to use this repo for our BA Project
#### STEP1 : Clone the repo and switch into the repo
```
git clone https://github.com/chandrasekar5195/BA_Project
cd BA_Project
```

#### STEP2 : Create a branch for you to work on 
```
#If you're creating the branch for the first time use "-b" 
git checkout -b chandra_branch

#If you've already created the branch and you're reworking on the same branch, use checkout without "-b"
git checkout chandra_branch
```

#### STEP3: Once you're happy with the changes, commit your changes to your branch
```
#To add all the files in your repo for the commit
git add . 
git commit -m "Added a new model to the code"
```

#### Step4: Once committed, push the code. 
```
To authenticate, your username will be your github username and your password will be the personal access token you generate. 
git push origin chandra_branch
```

#### Step5: Once you've pushed your changes to your branch, we can discuss and merge it with the main branch

