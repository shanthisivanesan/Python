import os
os.getcwd()

# slide 2
# Read a text file
countries = open("data/countries.txt", "r")
# Print out the results
for line in countries:
    line

# Close when done
countries.close()
# slide 3
## Use with to open files to ensure closure 
with open(r"data/green_eggs_ham.txt") as f1: 
    dat = f1.read() 
dat
# slide 4
## Process files one line at a time
with open(r"data/green_eggs_ham.txt") as f1: 
    dat = f1.readline() 
    dat = f1.readline() 
    dat = f1.readline() 
dat

## read in lines as a list of strings 
with open(r"data/green_eggs_ham.txt") as f1: 
    dat1 = f1.readlines() 
#dat1
# slide 5
# Open a text file for writing
scores = open("data/scores.txt", "w")
# write down some scores
while True:
    participant = input("Participant name > ")
    if participant == "quit":
        print("Quitting...")
        break 
    
    score = input("Score for " + participant + " > ") 
    scores.write(participant + ", " + score + "\n")

# Close when done
scores.close()