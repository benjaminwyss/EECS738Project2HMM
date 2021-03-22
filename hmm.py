import numpy as np
import random

class hmm:

	def __init__(self):
		self.X = None
		pass

	def fit(self, X):
		self.X = X

		# Use a dictionary to map words to states and vice versa
		self.wordStateDict = {"<start>": 0, "<end>": 1}
		self.stateWordDict = ["<start>", "<end>"]
		self.uniqueWords = 2
		
		# Determine all unique words in text corpus X
		for line in self.X:
			for word in line[0].split():
				if word not in self.wordStateDict:
					self.wordStateDict[word] = self.uniqueWords
					self.stateWordDict.append(word)
					self.uniqueWords += 1

		# Initialize state transition matrix
		self.transitionMatrix = np.zeros((self.uniqueWords, self.uniqueWords))
		self.allStates = range(self.uniqueWords)
		
		# Determine probability of state transitions
		for line in self.X:
			prevWord = "<start>"
			for word in line[0].split():
				if word == "<start>":
					continue
				prevState = self.wordStateDict[prevWord]
				currState = self.wordStateDict[word]
				self.transitionMatrix[prevState, currState] += 1
				prevWord = word
		self.transitionMatrix = self.transitionMatrix/self.transitionMatrix.sum(axis=1, keepdims=1)

		
	def generateFakeText(self):
		# Start at begin state
		currState = 0
		fakeText = ""
		while True:
			# Randomly determine next state based on transition matrix
			nextState = np.random.choice(self.allStates, 1, p=self.transitionMatrix[currState])[0]
			
			# Break if the next state is the end state
			if nextState == 1:
				break

			# Otherwise, add the next state's word to the fake text and update current state
			fakeText += self.stateWordDict[nextState] + " "
			currState = nextState

		return fakeText

	def predictText(self, prompt):
		# Clean prompt text
		cleanText = ""
		for c in prompt:
			if c in ["'", '"', ':', '‘', '’', '.', ',', '?', '!', '“', '”', ';', '(', ')', '[', ']', '`', 'â', '\x80', '\x99', '\u200b', '·']:
				continue
			elif c in ['—', '-', '–', '/', '\\', '|']:
				cleanText += ' '
			elif c == '%':
				cleanText += ' percent'
			else:
				cleanText += c
			
		prompt = cleanText.lower()

		# Determine current state from the prompt
		lastWord = prompt.split()[-1]
		
		if lastWord not in self.wordStateDict:
			return "Error. Prompt contains words not learned by the Hidden Markov Model."

		predictedText = ""
		currState = self.wordStateDict[lastWord]
		while True:
			# Determine most likely next state
			nextState = np.argmax(self.transitionMatrix[currState])

			# Break if the next state is the end state or if a word is repeated (this second condition avoids infinite loops)
			if nextState == 1 or self.stateWordDict[nextState] in predictedText:
				break

			# Otherwise, add the next state's word to the fake text and update current state
			predictedText += " " + self.stateWordDict[nextState]
			currState = nextState

		return prompt + predictedText