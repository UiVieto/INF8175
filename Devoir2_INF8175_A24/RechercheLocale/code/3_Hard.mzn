% Huy Viet Nguyen (2136374)
% Alaa Eddine Chenak (1976567)

include "globals.mzn"; 

%-----------------------------------------------------------------------------%
% Données
%-----------------------------------------------------------------------------%

% Nombre de villes à visiter + point de départ de la tournée
int: n;
% Distance entre chaque villes (symmétrique)
array[1..n,1..n] of int: distance;
% Distance minimale possible entre deux villes différentes
int: min_val = min([distance[i,j] | i,j in 1..n where distance[i,j] > 0]);
% Distance maximale possible entre deux villes différentes
int: max_val = max([distance[i,j] | i,j in 1..n]);

% Fenêtre de temps (1 = début, 2 = fin) durant laquelle il est possible de jouer pour chaque ville
array[1..n,1..2] of int: fenetre_arrivee;
% Temps maximum pour tout le trajet
int: max_trajet = max([fenetre_arrivee[i,2] | i in 1..n]);
% Fenêtre de temps la plus grande du trajet
int: max_fenetre = max([fenetre_arrivee[i,2] - fenetre_arrivee[i,1]| i in 2..n]);

%-----------------------------------------------------------------------------%
% Variables de décisions
%-----------------------------------------------------------------------------%

% Ordre des villes visitées par la tournée
array[1..n] of var 1..n: chemin_tournee;
% Circuit de la tournée, circuit_tournee[i] = j => j est le successeur de i
array[1..n] of var 1..n: circuit_tournee;
% Temps cumulé lors de l'arrivé au point i de la tournée
array[1..n] of var 0..max_trajet: cumul_temps;
% Temps d'attente avant le spectacle en ville i
array[1..n] of var 0..max_fenetre: temps_attente;
% Temps d'attente total 
var 0..max_trajet: total_temps_attentes = sum(temps_attente);

%-----------------------------------------------------------------------------%
% Objectif
%-----------------------------------------------------------------------------%

% Temps de trajet à minimiser
var int: temps_trajet;

%-----------------------------------------------------------------------------%
% Partie étudiant
%-----------------------------------------------------------------------------%
% Le temps du trajet ne peut pas dépasser le temps maximal pour un trajet
constraint temps_trajet <= max_trajet;

% Le trajet doit passer par tous les villes dans la fenêtre d'arrivée
constraint forall(i in 1..n)(cumul_temps[i] <= fenetre_arrivee[chemin_tournee[i], 2]);

% Les temps cumulés et le temps du trajet sont toujours bornés par un facteur de la distance minimale entre deux villes
constraint forall(i in 1..n)(cumul_temps[i] >= min_val * i);
constraint temps_trajet >= min_val  * n;

% Le temps d'attente ne peut pas être plus grand que la fenetre
constraint forall(i in 1..n)(temps_attente[chemin_tournee[i]] <= fenetre_arrivee[chemin_tournee[i], 2] - fenetre_arrivee[chemin_tournee[i], 1]);
constraint forall(i in 1..n)(temps_attente[i] >= 0);

% La tournée doit passer par une ville une fois. Et, la tournée commence et se termine à la ville 1.
constraint all_different(chemin_tournee);
constraint all_different(circuit_tournee);
constraint chemin_tournee[n] = 1;

% Circuit
constraint circuit_tournee[1] = chemin_tournee[1];
constraint forall(i in 2..n)(circuit_tournee[chemin_tournee[i-1]] = chemin_tournee[i]);

% Entrée dans la première ville
constraint cumul_temps[1] = max(fenetre_arrivee[chemin_tournee[1], 1], distance[1, chemin_tournee[1]]);
constraint temps_attente[chemin_tournee[1]] = max(fenetre_arrivee[chemin_tournee[1], 1] - distance[1, chemin_tournee[1]], 0);

% Le reste des villes
constraint forall(i in 2..n)
    (cumul_temps[i] = max(cumul_temps[i-1] + distance[chemin_tournee[i-1], chemin_tournee[i]], fenetre_arrivee[chemin_tournee[i], 1]));
constraint forall(i in 2..n)
    (temps_attente[chemin_tournee[i]] = max(fenetre_arrivee[chemin_tournee[i], 1] - cumul_temps[i-1] - distance[chemin_tournee[i-1], chemin_tournee[i]], 0));

temps_trajet = cumul_temps[n];

solve minimize temps_trajet;
%-----------------------------------------------------------------------------%

output [
  "Chemin de la tournée                   : \(chemin_tournee)\n",
  "Circuit de la tournée                  : \(circuit_tournee)\n",
  "Temps cumulés avant d'entrer en ville  : \(cumul_temps)\n",
  "Temps d'attente avant l'entrée en ville: \(temps_attente)\n",
  "Temps d'attente total                  : \(total_temps_attentes)\n",
  "Temps de trajet total                  : \(temps_trajet)\n",
];