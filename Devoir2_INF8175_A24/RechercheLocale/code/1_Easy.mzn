% Huy Viet Nguyen (2136374)
% Alaa Eddine Chenak (1976567)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre d'acteurs et de rôles
int: n;
% Couleurs des costumes
enum costumes; 
% Costume démandé pour un rôle i (première dimension) par un acteur j (seconde dimension)
array[1..n,1..n] of costumes: choix_costume;

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Variables de décision pour assigner un acteur à un rôle
array [1..n] of var 1..n: assignation_role;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
constraint all_different(i in 1..n)(choix_costume[assignation_role[i], i]);
constraint all_different(i in 1..n)(assignation_role[i]);
constraint forall(i in 2..n-1)(assignation_role[i] != (assignation_role[i+1] + 1));
constraint forall(i in 2..n-1)(assignation_role[i] != (assignation_role[i+1] - 1));
constraint forall(i in 2..n-1)(assignation_role[i] != (assignation_role[i-1] + 1));
constraint forall(i in 2..n-1)(assignation_role[i] != (assignation_role[i-1] - 1));
constraint assignation_role[1] != (assignation_role[2] + 1);
constraint assignation_role[n] != (assignation_role[n-1] - 1);

solve satisfy;
%-----------------------------------------------------------------------------%

output	
  ["L'acteur " ++ format(i) ++ 
   " prendra le rôle " ++ format(fix(assignation_role[i])) ++ 
   " (costume " ++ format(choix_costume[fix(assignation_role[i]), i]) ++ ")" ++"\n" 
  | i in 1..n];