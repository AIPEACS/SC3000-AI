% ======================================================
% Exercise 1: The Smart Phone Rivalry
% CZ3005 Artificial Intelligence - Assignment 2
% ======================================================
%
/* --- Part 1.1: First Order Logic (FOL) Statements ---

   1.  Competitor(sumsum, appy)
         "sumsum is a competitor of appy"

   2.  Developed(sumsum, galactica_s3)
         "sumsum developed galactica_s3"

   3.  Smartphone(galactica_s3)
         "galactica_s3 is smart phone technology"

   4.  Stole(stevey, galactica_s3)
         "stevey stole galactica_s3"

   5.  Boss(stevey, appy)
         "stevey is a boss of appy"

   6.  ∀x∀y [ Competitor(x, y) → Rival(x, y) ]
         "a competitor is a rival"

   7.  ∀x [ Smartphone(x) → Business(x) ]
         "smart phone technology is business"

   8.  ∀x∀y∀z∀w [ Boss(x,y) ∧ Stole(x,z) ∧ Business(z)
                   ∧ Developed(w,z) ∧ Rival(w,y)
                   → Unethical(x) ]
         "it is unethical for a boss to steal business
          from rival companies"
*/
%
% --- Part 1.2: Prolog Clauses ---

% --- Facts ---

% sumsum is a competitor of appy
competitor(sumsum, appy).

% sumsum developed galactica_s3
developed(sumsum, galactica_s3).

% galactica_s3 is smart phone technology
smartphone(galactica_s3).

% stevey stole galactica_s3
stole(stevey, galactica_s3).

% stevey is a boss of appy
boss(stevey, appy).

% --- Rules ---

% A competitor is a rival (FOL 6)
rival(X, Y) :- competitor(X, Y).

% Smart phone technology is business (FOL 7)
business(X) :- smartphone(X).

% It is unethical for a boss to steal business from rival companies (FOL 8)
% X is unethical if:
%   X is a boss of company Y,
%   X stole item Z,
%   Z is business,
%   W developed Z,
%   W is a rival of Y (i.e. W is a rival company to Y)
unethical(X) :-
    boss(X, Y),
    stole(X, Z),
    business(Z),
    developed(W, Z),
    rival(W, Y).

% --- Part 1.3: Query to prove Stevey is unethical ---
% Run in SWI-Prolog:
%   ?- trace.
%   ?- unethical(stevey).
