#Setup and build class

import collections

Card = collections.namedtuple('Card',['rank','suit']) #define what a card is

#Deck class
class FrenchDeck:
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds hearts clubs'.split()
    
    def __init__(self):
        self._cards = [Card(rank, suit) 
                       for suit in self.suits 
                       for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
        
#Pick a random card
from random import choice
choice(deck)

#Testing Sorting
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit] #We can now list deck in order of increasing rank
    
for card in sorted(deck, key=spades_high):
    print(card)
