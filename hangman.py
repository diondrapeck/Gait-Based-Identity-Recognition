## Game of Hangman designed for three preset categories - a total of 16 possible
## words. Shows visual Hangman representation as game progresses. 7 wrong 
## guesses allowed before game end.

import random as rand

# returns list of indices of substring indices
def findall(obj, value, start=-1):
    lst = []
    
    if value in obj:
        for element in obj:
            if element == value:
                lst.append(obj.index(value, start + 1))
                start = obj.index(value, start + 1)
    return lst

# draws "hangman" based on number of wrong guesses
def hangman(num):
    tree = " ___\n/\n|\n|\n|\n|\n"
    head = " ___\n/  O\n|\n|\n|\n|\n"
    top = " ___\n/  O\n|  |\n|\n|\n|\n"
    one_arm = " ___\n/  O\n| \|\n|\n|\n|\n"
    both_arms = " ___\n/  O\n| \|/\n|\n|\n|\n"
    bottom = " ___\n/  O\n| \|/\n|  |\n|\n|\n"
    one_leg = " ___\n/  O\n| \|/\n|  |\n| / \n|\n"
    both_legs = " ___\n/  O\n| \|/\n|  |\n| / \ \n|\n"

    bodyparts = {0: tree, 1: head, 2: top, 3: one_arm, 4: both_arms, 5: bottom,
    6: one_leg, 7: both_legs}
    
    print bodyparts[num]
    
print "Welcome to Hangman! Try to guess the mystery word before being hanged."

theme = ""
while not theme:
    theme = raw_input("Choose a theme for your mystery word: Board Games, Music\
 Artists, or Programming Languages\n")

    if theme == "Board Games":
        wordlist = ["LIFE", "TROUBLE", "MONOPOLY", "SCRABBLE", "SORRY"]
    elif theme == "Music Artists":
        wordlist = ["PRINCE", "BEYONCE", "QUEEN", "MADONNA", "ADELE"]
    elif theme == "Programming Languages":
        wordlist = ["PERL", "JAVA", "PYTHON", "HASKELL", "FORTRAN", "OCAML"]
    else:
        print "Please enter one of the provided theme options to continue."
        theme = ""
        continue

word = rand.choice(wordlist)
letters = len(word)
wrong_guesses = 0
correct_guesses = 0
display = "_" * letters
print display

while correct_guesses < letters and wrong_guesses < 7:
        guess = (raw_input("Guess a letter: \n")).upper()
        if guess in word:
            for letter in findall(word, guess):
                display = display[:letter] + guess + display[(letter + 1):]
                correct_guesses += 1
        else:
            wrong_guesses += 1
            print "Sorry, %c is not in the mystery word. Try again!" % guess 
        hangman(wrong_guesses)  
        print display

if display == word:
    print "\n Congratulations! You solved the mystery word! :)"
else:
    print "\n Oh no! You've used up all of your lives. The mystery word was: %s . :( Better luck next time!" % word