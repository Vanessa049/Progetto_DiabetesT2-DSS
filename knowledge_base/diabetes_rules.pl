/*
 * Knowledge Base Prolog per Diagnosi Diabete Tipo 2
 */

/* ============================================================
   FATTI DINAMICI (caricati da Python)
   ============================================================ */

:- dynamic paziente/5.       % paziente(ID, Eta, Gravidanze, Pedigree, NumRiskFactors)
:- dynamic esame/3.          % esame(PazID, TipoEsame, Valore)

/* ============================================================
   REGOLE BASE - CLASSIFICAZIONE PER ETA'
   ============================================================ */

fascia_eta(PazID, giovane) :-
    paziente(PazID, Eta, _, _, _),
    Eta < 35.

fascia_eta(PazID, medio) :-
    paziente(PazID, Eta, _, _, _),
    Eta >= 35, Eta < 45.

fascia_eta(PazID, rischio) :-
    paziente(PazID, Eta, _, _, _),
    Eta >= 45, Eta < 60.

fascia_eta(PazID, anziano) :-
    paziente(PazID, Eta, _, _, _),
    Eta >= 60.

/* ============================================================
   REGOLE CLINICHE - SOGLIE METABOLICHE (linee guida ADA 2023)
   ============================================================ */

% --- Glicemia a digiuno (mg/dL) ---
glicemia_normale(PazID) :-
    esame(PazID, glucose, Val), Val > 0, Val < 100.

glicemia_prediabete(PazID) :-
    esame(PazID, glucose, Val), Val >= 100, Val < 126.

glicemia_diabete(PazID) :-
    esame(PazID, glucose, Val), Val >= 126.

% --- BMI (kg/m^2) secondo WHO ---
bmi_normale(PazID) :-
    esame(PazID, bmi, Val), Val > 0, Val < 25.0.

bmi_sovrappeso(PazID) :-
    esame(PazID, bmi, Val), Val >= 25.0, Val < 30.0.

bmi_obeso(PazID) :-
    esame(PazID, bmi, Val), Val >= 30.0.

% Obesita' severa (classe II+): rischio molto elevato
bmi_obeso_severo(PazID) :-
    esame(PazID, bmi, Val), Val >= 35.0.

% --- Pressione diastolica (mmHg) secondo JNC 8 ---
pressione_normale(PazID) :-
    esame(PazID, blood_pressure, Val), Val > 0, Val < 80.

pressione_alta(PazID) :-
    esame(PazID, blood_pressure, Val), Val >= 80, Val < 90.

ipertensione_diastolica(PazID) :-
    esame(PazID, blood_pressure, Val), Val >= 90.

% --- Insulina sierica (mu U/ml) ---
insulina_alta(PazID) :-
    esame(PazID, insulin, Val), Val > 130.

insulino_resistenza_probabile(PazID) :-
    bmi_obeso(PazID),
    insulina_alta(PazID).

% --- Spessore plica cutanea (tricipite, mm) ---
adiposita_viscerale(PazID) :-
    esame(PazID, skin_thickness, Val), Val > 30.

% --- Pedigree diabetico (DiabetesPedigreeFunction) ---
storia_familiare_forte(PazID) :-
    paziente(PazID, _, _, Pedigree, _),
    Pedigree > 0.8.

storia_familiare_moderata(PazID) :-
    paziente(PazID, _, _, Pedigree, _),
    Pedigree > 0.4, Pedigree =< 0.8.

% --- Gravidanze multiple (fattore di rischio GDM -> T2DM) ---
rischio_gdm(PazID) :-
    paziente(PazID, _, Gravidanze, _, _),
    Gravidanze >= 4.

/* ============================================================
   CORRELAZIONI TRA FATTORI DI RISCHIO METABOLICI
   (background knowledge medico - ADA 2023)
   ============================================================ */

correlazione(obesita, insulino_resistenza).
correlazione(insulino_resistenza, obesita).
correlazione(obesita, ipertensione).
correlazione(ipertensione, obesita).
correlazione(obesita, glicemia_alta).
correlazione(glicemia_alta, obesita).
correlazione(insulino_resistenza, glicemia_alta).
correlazione(glicemia_alta, insulino_resistenza).
correlazione(eta_avanzata, insulino_resistenza).
correlazione(eta_avanzata, ipertensione).
correlazione(storia_familiare, rischio_genetico).
correlazione(gdm, diabete_futuro).
correlazione(ipertensione, sindrome_metabolica).
correlazione(obesita, sindrome_metabolica).

fattori_correlati(F1, F2) :-
    correlazione(F1, F2).

/* ============================================================
   CLASSIFICAZIONE DEL RISCHIO DIABETICO
   (inferenza multi-livello basata su linee guida ADA 2023)
   ============================================================ */

% Rischio CRITICO: glicemia gia' da diabete + almeno un fattore aggravante
rischio_critico(PazID) :-
    glicemia_diabete(PazID),
    bmi_obeso(PazID).

rischio_critico(PazID) :-
    glicemia_diabete(PazID),
    fascia_eta(PazID, anziano).

rischio_critico(PazID) :-
    glicemia_diabete(PazID),
    insulino_resistenza_probabile(PazID).

% Rischio ALTO: pre-diabete + fattori di rischio multipli, o obesita' severa
rischio_alto(PazID) :-
    \+ rischio_critico(PazID),
    glicemia_prediabete(PazID),
    bmi_obeso(PazID),
    fascia_eta(PazID, F),
    (F = rischio ; F = anziano).

rischio_alto(PazID) :-
    \+ rischio_critico(PazID),
    bmi_obeso_severo(PazID),
    ipertensione_diastolica(PazID).

rischio_alto(PazID) :-
    \+ rischio_critico(PazID),
    paziente(PazID, _, _, _, NumRisk),
    NumRisk >= 4.

rischio_alto(PazID) :-
    \+ rischio_critico(PazID),
    glicemia_prediabete(PazID),
    storia_familiare_forte(PazID),
    bmi_sovrappeso(PazID).

% Rischio MODERATO: pre-diabete senza obesita', o BMI obeso senza iperglicemia
rischio_moderato(PazID) :-
    \+ rischio_critico(PazID),
    \+ rischio_alto(PazID),
    glicemia_prediabete(PazID).

rischio_moderato(PazID) :-
    \+ rischio_critico(PazID),
    \+ rischio_alto(PazID),
    bmi_obeso(PazID),
    fascia_eta(PazID, F),
    (F = rischio ; F = anziano).

rischio_moderato(PazID) :-
    \+ rischio_critico(PazID),
    \+ rischio_alto(PazID),
    paziente(PazID, _, _, _, NumRisk),
    NumRisk >= 2, NumRisk < 4.

% Rischio BASSO
rischio_basso(PazID) :-
    \+ rischio_critico(PazID),
    \+ rischio_alto(PazID),
    \+ rischio_moderato(PazID).

% Livello rischio unificato (con cut per evitare backtracking multiplo)
livello_rischio(PazID, critico)  :- rischio_critico(PazID), !.
livello_rischio(PazID, alto)     :- rischio_alto(PazID), !.
livello_rischio(PazID, moderato) :- rischio_moderato(PazID), !.
livello_rischio(PazID, basso)    :- rischio_basso(PazID).

/* ============================================================
   RACCOMANDAZIONI CLINICHE
   (inferenza multi-livello: fattori -> rischio -> piano)
   ============================================================ */

% Piano intervento per rischio CRITICO
raccomanda_intervento(PazID, ogtt_immediato) :-
    rischio_critico(PazID).

raccomanda_intervento(PazID, hba1c_trimestrale) :-
    rischio_critico(PazID).

raccomanda_intervento(PazID, consulenza_diabetologica_urgente) :-
    rischio_critico(PazID).

% Piano intervento per rischio ALTO
raccomanda_intervento(PazID, hba1c_semestrale) :-
    rischio_alto(PazID),
    \+ rischio_critico(PazID).

raccomanda_intervento(PazID, programma_prevenzione_diabete) :-
    rischio_alto(PazID).

raccomanda_intervento(PazID, consulenza_nutrizionale) :-
    rischio_alto(PazID),
    bmi_obeso(PazID).

% Piano intervento per rischio MODERATO
raccomanda_intervento(PazID, glicemia_annuale) :-
    rischio_moderato(PazID),
    \+ rischio_alto(PazID).

raccomanda_intervento(PazID, modifica_stile_vita) :-
    rischio_moderato(PazID).

% Controllo standard per rischio BASSO
raccomanda_intervento(PazID, screening_triennale) :-
    rischio_basso(PazID).

/* ============================================================
   DIAGNOSI ABDUTTIVA
   (spiega PERCHE' un paziente ha un certo livello di rischio)
   ============================================================ */

diagnosi_rischio(PazID, glicemia_diabetica(Val)) :-
    esame(PazID, glucose, Val), Val >= 126.

diagnosi_rischio(PazID, glicemia_prediabetica(Val)) :-
    esame(PazID, glucose, Val), Val >= 100, Val < 126.

diagnosi_rischio(PazID, obesita(Val)) :-
    esame(PazID, bmi, Val), Val >= 30.

diagnosi_rischio(PazID, obesita_severa(Val)) :-
    esame(PazID, bmi, Val), Val >= 35.

diagnosi_rischio(PazID, ipertensione(Val)) :-
    esame(PazID, blood_pressure, Val), Val >= 90.

diagnosi_rischio(PazID, insulino_resistenza) :-
    insulino_resistenza_probabile(PazID).

diagnosi_rischio(PazID, adiposita_viscerale) :-
    adiposita_viscerale(PazID).

diagnosi_rischio(PazID, eta_avanzata) :-
    paziente(PazID, Eta, _, _, _), Eta >= 45.

diagnosi_rischio(PazID, storia_familiare_forte) :-
    storia_familiare_forte(PazID).

diagnosi_rischio(PazID, storia_familiare_moderata) :-
    storia_familiare_moderata(PazID).

diagnosi_rischio(PazID, rischio_gravidanze_multiple) :-
    rischio_gdm(PazID).

/* ============================================================
   PROFILO TIPICO - PATTERN RECOGNITION BASATO SU REGOLE
   (cattura il profilo clinico classico del paziente diabetico T2)
   ============================================================ */

% Profilo classico T2DM: iperglicemia + obesita' + eta' a rischio
profilo_tipico_t2dm(PazID) :-
    glicemia_prediabete(PazID),
    bmi_obeso(PazID),
    fascia_eta(PazID, F),
    (F = rischio ; F = anziano).

profilo_tipico_t2dm(PazID) :-
    glicemia_diabete(PazID),
    bmi_sovrappeso(PazID).

profilo_tipico_t2dm(PazID) :-
    insulino_resistenza_probabile(PazID),
    ipertensione_diastolica(PazID).

% Sindrome metabolica: obesita' + ipertensione + insulino-resistenza
sindrome_metabolica(PazID) :-
    bmi_obeso(PazID),
    ipertensione_diastolica(PazID),
    insulina_alta(PazID).

/* ============================================================
   LIVELLO SCREENING NUMERICO
   (catena inferenziale: fattori -> rischio -> livello 0-3)
   ============================================================ */

% screening_level: 0=triennale, 1=annuale, 2=semestrale, 3=trimestrale/urgente
screening_level(PazID, 3) :- rischio_critico(PazID), !.
screening_level(PazID, 2) :- rischio_alto(PazID), !.
screening_level(PazID, 1) :- rischio_moderato(PazID), !.
screening_level(_, 0).

/* ============================================================
   FATTORI DIAGNOSTICI INDIVIDUALI (per conteggio abduttivo)
   ============================================================ */

ha_fattore_diag(PazID, glicemia_diabete)      :- glicemia_diabete(PazID).
ha_fattore_diag(PazID, glicemia_prediabete)   :- glicemia_prediabete(PazID).
ha_fattore_diag(PazID, obesita)               :- bmi_obeso(PazID).
ha_fattore_diag(PazID, sovrappeso)            :- bmi_sovrappeso(PazID).
ha_fattore_diag(PazID, ipertensione)          :- ipertensione_diastolica(PazID).
ha_fattore_diag(PazID, insulino_resistenza)   :- insulino_resistenza_probabile(PazID).
ha_fattore_diag(PazID, adiposita_viscerale)   :- adiposita_viscerale(PazID).
ha_fattore_diag(PazID, eta_rischio)           :- paziente(PazID, Eta, _, _, _), Eta >= 45.
ha_fattore_diag(PazID, storia_familiare)      :- storia_familiare_forte(PazID).
ha_fattore_diag(PazID, rischio_gravidanze)    :- rischio_gdm(PazID).

conta_diagnosi(PazID, N) :-
    findall(F, ha_fattore_diag(PazID, F), Fattori),
    length(Fattori, N).

/* ============================================================
   COPPIE DI FATTORI CORRELATI CO-PRESENTI
   (cattura interazioni metaboliche note come background knowledge)
   ============================================================ */

coppia_presente(PazID, obesita_ipertensione) :-
    bmi_obeso(PazID), ipertensione_diastolica(PazID).

coppia_presente(PazID, obesita_prediabete) :-
    bmi_obeso(PazID), glicemia_prediabete(PazID).

coppia_presente(PazID, obesita_insulino_resistenza) :-
    bmi_obeso(PazID), insulina_alta(PazID).

coppia_presente(PazID, eta_iperglicemia) :-
    paziente(PazID, Eta, _, _, _), Eta >= 45,
    (glicemia_prediabete(PazID) ; glicemia_diabete(PazID)).

coppia_presente(PazID, ipertensione_insulino_resistenza) :-
    ipertensione_diastolica(PazID), insulina_alta(PazID).

coppia_presente(PazID, storia_familiare_obesita) :-
    storia_familiare_forte(PazID), bmi_obeso(PazID).

conta_coppie_correlate(PazID, N) :-
    findall(C, coppia_presente(PazID, C), Coppie),
    length(Coppie, N).

/* ============================================================
   FEATURE DERIVATE PER IL MACHINE LEARNING
   (integrazione reale tra ragionamento Prolog e apprendimento ML)
   ============================================================ */

% kb_screening_level (0-3): livello di screening raccomandato
% Codifica la catena inferenziale completa: dati -> fattori -> rischio -> screening
is_screening_level(PazID, Level) :- screening_level(PazID, Level).

% kb_num_diagnosi (0-10): quanti fattori dalla diagnosi abduttiva
% Conta l'output dell'abduzione Prolog come feature numerica
is_num_diagnosi(PazID, N) :- conta_diagnosi(PazID, N).

% kb_fattori_correlati (0-6): coppie di fattori metabolici co-presenti
% Cattura interazioni che ML fatica a imparare da dati sparsi
is_num_coppie_correlate(PazID, N) :- conta_coppie_correlate(PazID, N).

% kb_profilo_tipico (0/1): pattern classico T2DM riconosciuto dalla KB
is_profilo_tipico(PazID, 1) :- profilo_tipico_t2dm(PazID), !.
is_profilo_tipico(_, 0).

/* ============================================================
   ESEMPI DI UTILIZZO
   ============================================================

   ?- livello_rischio(1, Livello).
   ?- diagnosi_rischio(1, Causa).
   ?- raccomanda_intervento(1, Tipo).
   ?- sindrome_metabolica(1).
   ?- fattori_correlati(obesita, X).
   ?- is_screening_level(1, L).

   ============================================================ */
