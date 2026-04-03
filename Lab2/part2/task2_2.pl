% NEW Royal Succession Rule - Modified British Monarchy System
% The throne is now passed down according to the order of birth
% irrespective of gender (male and female treated equally).

% Family relationships
offspring(elizabeth, charles, 1).
offspring(elizabeth, anne, 2).
offspring(elizabeth, andrew, 3).
offspring(elizabeth, edward, 4).

% Gender of each person
male(charles).
male(andrew).
male(edward).
male(william).
male(henry).

female(elizabeth).
female(anne).

% Determine if a person is male
is_male(X) :- male(X).

% Determine if a person is female
is_female(X) :- female(X).

% Get all offspring of a parent in birth order
get_offspring(Parent, Child, BirthOrder) :-
    offspring(Parent, Child, BirthOrder).

% Get male offspring of a parent
male_offspring(Parent, Child) :-
    offspring(Parent, Child, _),
    is_male(Child).

% Get female offspring of a parent
female_offspring(Parent, Child) :-
    offspring(Parent, Child, _),
    is_female(Child).

% NEW Succession rule: determine next in line for a given parent
% All offspring in birth order, regardless of gender
successor(Parent, Successor) :-
    offspring(Parent, Successor, BirthOrder1),
    \+ (offspring(Parent, _, BirthOrder2),
        BirthOrder2 < BirthOrder1).

% NEW Line of succession: returns entire line of succession as a list
% All offspring sorted by birth order, irrespective of gender
line_of_succession(elizabeth, SuccessionList) :-
    findall(BirthOrder-X, offspring(elizabeth, X, BirthOrder), AllHeirs),
    sort(AllHeirs, SortedHeirs),
    pairs_values(SortedHeirs, SuccessionList).
