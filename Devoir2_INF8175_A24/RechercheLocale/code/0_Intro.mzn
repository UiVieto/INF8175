% Huy Viet Nguyen (2136374)
% Alaa Eddine Chenak (1976567)

include "globals.mzn";

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Nombre devant satisfaire les critères
var 1000..9999: nombre;

% Décomposition du nombre en un ensemble de chiffres
array[1..4] of var 1..9: chiffres;

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Variable à minimiser
var int: p;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
var int: objective = chiffres[2] * chiffres[3] * chiffres[4];

nombre = 1000*chiffres[1] + 100*chiffres[2] + 10*chiffres[3] + chiffres[4];

constraint nombre mod 2 = 0;
constraint all_different(chiffres);
constraint forall(k in 1..4)(chiffres[k] != 0);
constraint sum(i in 1..4)(chiffres[i]) > 15;
constraint chiffres[1] > chiffres[2];
constraint chiffres[3] < chiffres[4];
constraint chiffres[2] > chiffres[4];

solve minimize objective;
%-----------------------------------------------------------------------------%

output [
  show(nombre)
];

