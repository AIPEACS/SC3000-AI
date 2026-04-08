% Royal Succession Rule - Old British Monarchy System
% The throne passes along the male line according to birth order,
% before consideration along the female line.

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

% Succession rule: determine next in line for a given parent
% First priority: male offspring in birth order
% Second priority: female offspring in birth order
successor(Parent, Successor) :-
    offspring(Parent, Successor, BirthOrder1),
    is_male(Successor),
    \+ (offspring(Parent, Other, BirthOrder2),
        is_male(Other),
        BirthOrder2 < BirthOrder1).

successor(Parent, Successor) :-
    offspring(Parent, Successor, BirthOrder1),
    is_female(Successor),
    % Check no male offspring available
    \+ (offspring(Parent, Male, _), is_male(Male)),
    % Check no earlier female offspring
    \+ (offspring(Parent, Other, BirthOrder2),
        is_female(Other),
        BirthOrder2 < BirthOrder1).

% Succession from Elizabeth: returns entire line of succession as a list
% Males first by birth order, then females by birth order
line_of_succession(elizabeth, SuccessionList) :-
    findall(BirthOrder-X, (offspring(elizabeth, X, BirthOrder), is_male(X)), MaleHeirs),
    sort(MaleHeirs, SortedMales),
    findall(BirthOrder-X, (offspring(elizabeth, X, BirthOrder), is_female(X)), FemaleHeirs),
    sort(FemaleHeirs, SortedFemales),
    pairs_values(SortedMales, Males),
    pairs_values(SortedFemales, Females),
    append(Males, Females, SuccessionList).


