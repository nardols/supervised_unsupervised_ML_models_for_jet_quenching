/*
 *  Program to analyse hepmc files and calculate jet shape variables.
 *     Output is stored a ROOT Tree (jetprops below)
 *
 *  Usage: analyze_hepmc_jet_shapes_constsub_eventwise_treeout [--chargedjets|--fulljets] [--nobkg] <infile> <outfilebase>
 *
 *  Argumemts/switches:
 *    --nobkg: do not subtract background (for pp and JEWEL without recoil)
 *    --fulljets|--chargedjets: use all particles, or only charged particles for jet finding.
 *  Authors: Marco van Leeuwen, Nikhef; Miguel Romao, LIP
 *  Subtraction code: Korinna Zapp, Lund
 *  Edited by: Leonardo Lima, IF USP, April 2025
 *
 */

#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"

#include "fastjet/Selector.hh"                           //.......... Background Sutraction event by event
#include "fastjet/tools/JetMedianBackgroundEstimator.hh" //.......... Background Sutraction event by event
//#include "include/tools/Subtractor.hh"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/ClusterSequenceAreaBase.hh"

#include "fastjet/tools/Subtractor.hh"
#include "ConstituentSubtractor/ConstituentSubtractor.hh"
#include "Nsubjettiness/Nsubjettiness.hh"
#include "RecursiveTools/SoftDrop.hh"
#include "RecursiveTools/ModifiedMassDropTagger.hh"
#include "Nsubjettiness/AxesDefinition.hh"
#include "Nsubjettiness/MeasureDefinition.hh"

// #include "fastjet/contrib/DynamicalGroomer.hh"

#include "dyGroomerJet.hh"
// #include "jetCollection.hh"

#include "TPDGCode.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TMath.h"
#include "TVector2.h"
#include "TVector3.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "THn.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;
#include "getopt.h"


// defaults, can be set with arguments --chargedjets --fulljets --nobkgsub --bkgsub
int do_bkg = 0; // 0: no subtraction; 1: only jet energy; 2: energy and shape
int charged_jets = 0;

static const int debug = 0;
// static const int charged_constituents = 1; // This does not work yet for jet shapes!
static const float ptcut = 0.0;     // GeV
static const float min_jet_pt = 10; // GeV; for shape histos
// Bkg sub loose parameters
static const float _dRloose = 0.6;
static const float _etaloose = 2.5;
static const float _ptminloose = 70.;
// ------------------------
// ------------------------
#include <map>

std::map<int, float> PDGCharges = {
    // ---------- Leptons ----------
    {11, -1}, {-11, +1},   // Electron / Positron
    {13, -1}, {-13, +1},   // Muon / Anti-muon
    {15, -1}, {-15, +1},   // Tau / Anti-tau
    {12,  0}, {-12,  0},   // Electron neutrino / Anti-electron neutrino
    {14,  0}, {-14,  0},   // Muon neutrino / Anti-muon neutrino
    {16,  0}, {-16,  0},   // Tau neutrino / Anti-tau neutrino

    // ---------- Quarks ----------
    {1, +2.0f/3}, {-1, -2.0f/3},   // Up
    {2, -1.0f/3}, {-2, +1.0f/3},   // Down
    {3, -1.0f/3}, {-3, +1.0f/3},   // Strange
    {4, +2.0f/3}, {-4, -2.0f/3},   // Charm
    {5, -1.0f/3}, {-5, +1.0f/3},   // Bottom
    {6, +2.0f/3}, {-6, -2.0f/3},   // Top

    // ---------- Gluons and Bosons ----------
    {21, 0},   // Gluon
    {22, 0},   // Photon
    {23, 0},   // Z boson
    {24, +1}, {-24, -1},   // W+ / W-
    {25, 0},   // Higgs boson

    // ---------- Mesons ----------
    {111, 0},                 // pi0
    {211, +1}, {-211, -1},    // pi+ / pi-
    {130, 0},                 // K_L
    {310, 0},                 // K_S
    {321, +1}, {-321, -1},    // K+ / K-
    {421, 0}, {-421, 0},      // D0 / anti-D0
    {411, +1}, {-411, -1},    // D+ / D-
    {431, +1}, {-431, -1},    // D_s+ / D_s-
    {511, 0}, {-511, 0},      // B0 / anti-B0
    {521, +1}, {-521, -1},    // B+ / B-
    {531, 0}, {-531, 0},      // B_s0 / anti-B_s0
    {541, +1}, {-541, -1},    // B_c+ / B_c-

    // ---------- Baryons ----------
    {2212, +1}, {-2212, -1},  // Proton / Anti-proton
    {2112, 0}, {-2112, 0},    // Neutron / Anti-neutron
    {3122, 0}, {-3122, 0},    // Lambda / Anti-lambda
    {3222, +1}, {-3222, -1},  // Sigma+ / Anti-Sigma+
    {3212, 0}, {-3212, 0},    // Sigma0 / Anti-Sigma0
    {3112, -1}, {-3112, +1},  // Sigma- / Anti-Sigma-
    {3312, -1}, {-3312, +1},  // Xi- / Anti-Xi-
    {3322, 0}, {-3322, 0},    // Xi0 / Anti-Xi0
    {3334, -1}, {-3334, +1},  // Omega- / Anti-Omega-

    // ---------- Common resonances ----------
    {443, 0},       // J/psi
    {553, 0},       // Upsilon(1S)
    {100553, 0},    // Upsilon(2S)
    {200553, 0},    // Upsilon(3S)
    {223, 0},       // omega
    {333, 0},       // phi
    {313, 0}, {-313, 0},      // K*0 / anti-K*0
    {323, +1}, {-323, -1},    // K*+ / K*-
    {413, +1}, {-413, -1},    // D*+ / D*-
    {423, 0}, {-423, 0},      // D*0 / anti-D*0
    {513, +1}, {-513, -1},    // B*+ / B*-
    {523, 0}, {-523, 0},      // B*0 / anti-B*0

    // ---------- Generic neutral ----------
    {0, 0}   // Ghost / error
};

//----------------------------------------
// classe JetTree implementada manualmente
//----------------------------------------
class JetTreeManual {
public:
    JetTreeManual(const fastjet::PseudoJet& jet, const fastjet::ClusterSequence* seq,
                  double zcut = 0.1, double beta = 0.0);

    bool has_structure() const;
    float z() const;
    float delta() const;
    float kperp() const;
    float m() const;
    JetTreeManual harder() const;

private:
    std::vector<std::pair<fastjet::PseudoJet, fastjet::PseudoJet>> history;
    const fastjet::ClusterSequence* clust_seq;
    size_t index;
    fastjet::PseudoJet current;
    double zcut;
    double beta;

    void reconstruct_tree(const fastjet::PseudoJet& jet);
};

JetTreeManual::JetTreeManual(const fastjet::PseudoJet& jet, const fastjet::ClusterSequence* seq,
                             double zcut_, double beta_)
    : clust_seq(seq), index(0), zcut(zcut_), beta(beta_) {
    reconstruct_tree(jet);
    if (!history.empty())
        current = history[0].first;
    else
        current = jet;
}

void JetTreeManual::reconstruct_tree(const fastjet::PseudoJet& jet) {
    fastjet::PseudoJet p1, p2;
    fastjet::PseudoJet current_jet = jet;

    while (clust_seq->has_parents(current_jet, p1, p2)) {
        history.emplace_back(p1, p2);
        current_jet = (p1.perp() > p2.perp()) ? p1 : p2;
    }
}

bool JetTreeManual::has_structure() const {
    return index < history.size();
}

float JetTreeManual::z() const {
    const auto& [p1, p2] = history[index];
    float pt_sum = p1.perp() + p2.perp();
    return pt_sum > 0 ? p2.perp() / pt_sum : 0;
}

float JetTreeManual::delta() const {
    const auto& [p1, p2] = history[index];
    return p1.delta_R(p2);
}

float JetTreeManual::kperp() const {
    const auto& [p1, p2] = history[index];
    float zval = z();
    float dR = delta();
    float pt_sum = p1.perp() + p2.perp();
    return zval * dR * pt_sum;
}

float JetTreeManual::m() const {
    return current.m();
}

JetTreeManual JetTreeManual::harder() const {
    JetTreeManual next = *this;
    if (next.index < next.history.size()) {
        const auto& [p1, p2] = next.history[next.index];
        next.current = (p1.perp() > p2.perp()) ? p1 : p2;
        next.index++;
    }
    return next;
}


//----------------------------------
// class PDGINFO
//----------------------------------

class PDGInfo : public fastjet::PseudoJet::UserInfoBase
{
public:
    int pdg_id;
    PDGInfo(int id) : pdg_id(id){};
};

int is_charged(const HepMC::GenParticle *part)
{
    int abs_kf = abs(part->pdg_id());

    if (abs_kf == 211 || abs_kf == 321 || abs_kf == 2212 || abs_kf == 11 || abs_kf == 13)
        return 1;
    else if (abs_kf != 22 && abs_kf != 111 && abs_kf != 130 && abs_kf != 2112 && abs_kf != 311 && abs_kf != 12 && abs_kf != 14 && abs_kf != 16)
        cout << " Unexpected particle: kf=" << abs_kf << endl;
    return 0;
}

float dphi(float phi1, float phi2)
{
    float dphi = phi1 - phi2;
    float pi = 3.14159;
    if (dphi < -pi)
        dphi += 2 * pi;
    if (dphi > pi)
        dphi -= 2 * pi;
    return dphi;
}

void getcharge(const fastjet::PseudoJet &jet, float &jetcharge, float kappa)
{
    jetcharge = 0.;
    Double_t num = 0.;
    Double_t den = 0.;

    if (!jet.has_constituents())
        return;

    for (auto part : jet.constituents())
    {
        den = den + part.perp(); 

        int _pdgid = 0;  // standard value

        if (part.has_user_info<PDGInfo>()) {
            _pdgid = part.user_info<PDGInfo>().pdg_id;
        } else {
            std::cerr << "Warning: particle has no user_info. Assuming pdgid=0." << std::endl;
        }

        if (PDGCharges.count(_pdgid) == 1)
        {
            num += PDGCharges[_pdgid] * pow(part.perp(), kappa);
        }
        else
        {
            cout << " Charge for PDG id " << _pdgid << " is not defined, using 0." << endl;
        }
    }
    den = pow(den, kappa);
    if (den > 0)
        jetcharge = num / den;
    else
        jetcharge = 0.;
}

void getmassangularities(const fastjet::PseudoJet &jet, float &rm, float &r2m, float &zs, float &z2m, float &rz, float &r2z)
{
    // if (!jet.has_constituents())
    //     return;
    rm = 0;
    r2m = 0;
    zs = 0;
    z2m = 0;
    rz = 0;
    r2z = 0;
    if (!jet.has_constituents())
        return;
    std::vector<fastjet::PseudoJet> constits = jet.constituents();
    for (UInt_t ic = 0; ic < constits.size(); ++ic)
    {
        Double_t dphi = constits[ic].phi() - jet.phi();
        if (dphi < -1. * TMath::Pi())
            dphi += TMath::TwoPi();
        if (dphi > TMath::Pi())
            dphi -= TMath::TwoPi();
        Double_t dr2 = (constits[ic].rapidity() - jet.rapidity()) * (constits[ic].rapidity() - jet.rapidity()) + dphi * dphi;
        Double_t dr = TMath::Sqrt(dr2);
        Double_t zfrag = constits[ic].perp() / jet.perp();
        rm += dr;
        r2m += dr * dr;
        zs += zfrag;
        z2m += zfrag * zfrag;
        rz += dr * zfrag;
        r2z += dr * dr * zfrag;
    }
    rm /= constits.size();
    r2m /= constits.size();
    z2m /= constits.size();
}

float pTD(const fastjet::PseudoJet &jet)
{
    if (!jet.has_constituents())
        return 0;
    Double_t den = 0;
    Double_t num = 0.;
    std::vector<fastjet::PseudoJet> constits = jet.constituents();
    for (UInt_t ic = 0; ic < constits.size(); ++ic)
    {
        num = num + constits[ic].perp() * constits[ic].perp();
        den = den + constits[ic].perp();
    }
    return TMath::Sqrt(num) / den;
}

// Background subtraction -------------------------------------------------------------
struct MyDist
{
    double dR;
    size_t ipart;
    size_t ighost;
};
struct My4Mom
{
    double pt;
    double mdelta;
    double phi;
    double y;
};
struct MyPart
{
    double pt;
    double mdelta;
    double phi;
    double y;
    int id;
};

bool isZero(float val, double tolerance = 1e-8)
{
    return fabs(val) < tolerance;
}
bool fuzzyEquals(float a, float b, double tolerance = 1e-5)
{
    const double absavg = (fabs(a) + fabs(b)) / 2.0;
    const double absdiff = fabs(a - b);
    const bool rtn = (isZero(a) && isZero(b)) || absdiff < tolerance * absavg;
    return rtn;
}

float sqr(float x)
{ // This function is to facilitate the refactor from Korina's
    return pow(x, 2.0);
}

bool MyDistComp(const MyDist &dist1, const MyDist &dist2)
{
    return dist1.dR < dist2.dR;
}
/// event-wise Constituent subtraction based on Particles
vector<fastjet::PseudoJet> ConstSubPartEvent(const vector<fastjet::PseudoJet> &particles, vector<fastjet::PseudoJet> &pjghosts)
{
    // sort constituents into two vectors: particles and ghosts (subtraction momenta)
    vector<fastjet::PseudoJet> subevent;
    vector<MyPart> parts;
    vector<MyPart> ghosts;

    for (auto p : particles)
    {
        if (fuzzyEquals(p.E(), 1e-6))
        {
            continue;
        }
        MyPart part;
        part.id = p.user_info<PDGInfo>().pdg_id;
        part.pt = p.pt();
        part.mdelta = sqrt(p.m2() + sqr(p.pt())) - p.pt();
        part.phi = p.phi();
        part.y = p.rapidity();
        parts.push_back(part);
    }

    for (auto pj : pjghosts)
    {

        MyPart ghost;
        ghost.id = pj.user_info<PDGInfo>().pdg_id;
        ghost.pt = pj.pt();
        ghost.mdelta = sqrt(pj.m2() + sqr(pj.pt())) - pj.pt();
        ghost.phi = pj.phi();
        ghost.y = pj.rapidity();
        ghosts.push_back(ghost);
    }

    // cout<<"number of particles: "<<parts.size()<<endl;
    // cout<<"number of ghosts: "<<ghosts.size()<<endl;

    // create list with all particle-ghosts distances and sort it
    vector<MyDist> dists;
    for (size_t i = 0; i < parts.size(); ++i)
    {
        for (size_t j = 0; j < ghosts.size(); ++j)
        {
            // cout<<"i, j = "<<i<<" "<<j<<endl;
            MyDist dist;
            double Deltaphi(abs(parts[i].phi - ghosts[j].phi));
            if (Deltaphi > M_PI)
                Deltaphi = 2. * M_PI - Deltaphi;
            dist.dR = sqrt(sqr(Deltaphi) + sqr(parts[i].y - ghosts[j].y));
            dist.ipart = i;
            dist.ighost = j;
            // cout<<"distance: "<<dist.dR<<endl;
            dists.push_back(dist);
        }
    }
    // cout<<"number of pairs: "<<dists.size()<<endl;
    // dists.sort(MyDistComp);
    std::sort(dists.begin(), dists.end(), MyDistComp);

    // go through all particle-ghost pairs and re-distribute momentum and mass
    for (vector<MyDist>::iterator liter = dists.begin(); liter != dists.end(); ++liter)
    {
        // cout<<"dealing with dist "<<liter->dR<<endl;
        if (liter->dR > _dRloose)
            break;
        size_t pnum = liter->ipart;
        size_t gnum = liter->ighost;
        double ptp = parts[pnum].pt;
        double ptg = ghosts[gnum].pt;
        // cout<<"pts: "<<ptp<<" vs "<<ptg<<endl;
        if (ptp > ptg)
        {
            parts[pnum].pt -= ptg;
            ghosts[gnum].pt = 0.;
        }
        else
        {
            ghosts[gnum].pt -= ptp;
            parts[pnum].pt = 0.;
        }
        double mdp = parts[pnum].mdelta;
        double mdg = ghosts[gnum].mdelta;
        // cout<<"masses: "<<mdp<<" vs "<<mdg<<endl;
        if (mdp > mdg)
        {
            parts[pnum].mdelta -= mdg;
            ghosts[gnum].mdelta = 0.;
        }
        else
        {
            ghosts[gnum].mdelta -= mdp;
            parts[pnum].mdelta = 0.;
        }
    }

    // sum up resulting 4-momenta to get subtracted jet momentum
    int nparts(0);
    for (size_t i = 0; i < parts.size(); ++i)
    {
        if (parts[i].pt > 0.)
        {
            fastjet::PseudoJet outpart(
                parts[i].pt * cos(parts[i].phi),
                parts[i].pt * sin(parts[i].phi),
                (parts[i].pt + parts[i].mdelta) * sinh(parts[i].y),
                (parts[i].pt + parts[i].mdelta) * cosh(parts[i].y));
            outpart.set_user_info(new PDGInfo(parts[i].id));
            subevent.push_back(outpart);
            nparts++;
        }
    }
    int nghost(0);
    for (size_t i = 0; i < ghosts.size(); ++i)
    {
        if (ghosts[i].pt > 0.)
        {
            fastjet::PseudoJet outpart(
                ghosts[i].pt * cos(ghosts[i].phi),
                ghosts[i].pt * sin(ghosts[i].phi),
                (ghosts[i].pt + ghosts[i].mdelta) * sinh(ghosts[i].y),
                (ghosts[i].pt + ghosts[i].mdelta) * cosh(ghosts[i].y));
            outpart.set_user_info(new PDGInfo(ghosts[i].id));
            subevent.push_back(outpart);
            nghost++;
        }
        // else cout<<ghosts[i].mdelta<<endl;
    }
    // cout<<nghost<<" / "<<ghosts.size()+1<<" vs. "<<nparts<<" / "<<parts.size()<<endl;
    return subevent;
}

int main(int argc, char **argv)
{
    //
    // Takes two arguments: infile (HEPMC format) outfile (base name, ROOT format)
    // additional options: --nobkg --chargedjet|--fulljet
    //

    srand(time(NULL));

    if (argc < 2)
    {
        cerr << "Need two arguments: infile outfile" << endl
             << "infile is HEPMC ascii format; outfile will be root format" << endl;
        cerr << "further option arguments: [--chargedjets|--fulljets] [--nobkg]" << endl;
        return 1;
    }

    const double jetR = 0.4;
    const double zcut = 0.1;
    const double beta = 0;

    // RecursiveSoftDrop
    int Nrsd = -1;
    // -----------------

    const double max_eta_jet = 2.0;
    const double max_eta_track = 2.5;

    fastjet::contrib::OnePass_WTA_KT_Axes axes;
    fastjet::contrib::UnnormalizedMeasure unormbeta(1.0);

    fastjet::contrib::Nsubjettiness nSub1(1, axes, unormbeta);
    fastjet::contrib::Nsubjettiness nSub2(2, axes, unormbeta);
    fastjet::contrib::Nsubjettiness nSub3(3, axes, unormbeta);
    fastjet::contrib::Nsubjettiness nSub4(4, axes, unormbeta);
    fastjet::contrib::Nsubjettiness nSub5(5, axes, unormbeta);

    int c;

    int nopt_parsed = 0;
    while (1)
    {

        static struct option long_options[] =
            {
                /* These options set a flag. */
                {"chargedjets", no_argument, &charged_jets, 1},
                {"fulljets", no_argument, &charged_jets, 0},
                {"nobkgsub", no_argument, &do_bkg, 0},
                {"bkgsub", no_argument, &do_bkg, 1},
                /* it is also possible to have options that do not directly set a flag
                 * Not used for now */
                {0, 0, 0, 0}};
        /* getopt_long stores the option index here. */
        int option_index = 1;
        c = getopt_long(argc, argv, "",
                        long_options, &option_index);
        // cout << "c " << c << " option_index " << option_index << endl;
        /* Detect the end of the options. */
        if (c == -1)
            break;
        nopt_parsed++;
    }

    /* Print any remaining command line arguments (not options). */
    nopt_parsed++;
    cout << "option_index " << nopt_parsed << endl;
    if (nopt_parsed + 2 > argc)
    {
        cerr << "Need two more arguments: infile outfile" << endl
             << "infile is HEPMC ascii format; outfile will be root format" << endl;
        return 1;
    }

    char *inname = argv[nopt_parsed];
    // specify an input file
    HepMC::IO_GenEvent ascii_in(inname, std::ios::in);

    // Make histos

    string outname(argv[nopt_parsed + 1]);
    if (charged_jets)
        outname.append("_charged");
    else
        outname.append("_full");
    if (do_bkg == 0)
        outname.append("_nobkgsub");
    else
        outname.append("_bkgsub");
    outname.append(".root");

    cout << "Input: " << inname << ", output " << outname << endl;

    TFile fout(outname.c_str(), "RECREATE");

    Int_t ievt = 0, ijet = 0;
    Float_t evwt = 0, jet_eta = 0, jet_rapidity = 0, jet_phi = 0, jet_pt = 0;
    Float_t zg = 0, Rg = 0, kg = 0, mass = 0, mz2 = 0, mr = 0, mr2 = 0, rz = 0, r2z = 0;
    Int_t nconst = 0, nSD = 0;

    // Further additions
    Float_t ptd = 0;
    Float_t jetcharge03 = 0;
    Float_t jetcharge05 = 0;
    Float_t jetcharge07 = 0;
    Float_t jetcharge10 = 0;
    // ----------------
    // Nsubjettiness
    Float_t tau1 = 0, tau2 = 0, tau3 = 0, tau4 = 0, tau5 = 0, tau2tau1 = 0, tau3tau2 = 0;
    // ----------------
    // Dynamical Grooming
    Float_t kappa_TD = 0;
    Float_t kappa_ktD = 0;
    Float_t kappa_zD = 0;

    Float_t zg_TD = 0;
    Float_t zg_ktD = 0;
    Float_t zg_zD = 0;

    Float_t deltaR_TD = 0;
    Float_t deltaR_ktD = 0;
    Float_t deltaR_zD = 0;
    //------------------

    // Jet Observables
    TTree *jetprops = new TTree("jetprops", "Jet properties");
    jetprops->Branch("ievt", &ievt, "ievt/I");
    jetprops->Branch("ijet", &ijet, "ijet/I");
    jetprops->Branch("evwt", &evwt, "evwt/F");
    jetprops->Branch("pt", &jet_pt, "pt/F");
    jetprops->Branch("eta", &jet_eta, "eta/F");
    jetprops->Branch("rapidity", &jet_rapidity, "rapidity/F");
    jetprops->Branch("phi", &jet_phi, "phi/F");
    jetprops->Branch("nconst", &nconst, "nconst/I");
    jetprops->Branch("zg", &zg, "zg/F");
    jetprops->Branch("Rg", &Rg, "Rg/F");
    jetprops->Branch("kg", &kg, "kg/F");
    jetprops->Branch("nSD", &nSD, "nSD/I");
    jetprops->Branch("mass", &mass, "mass/F");
    jetprops->Branch("mz2", &mz2, "mz2/F");
    jetprops->Branch("mr", &mr, "mr/F");
    jetprops->Branch("mr2", &mr2, "mr2/F");
    jetprops->Branch("rz", &rz, "rz/F");
    jetprops->Branch("r2z", &r2z, "r2z/F");
    // ----------------
    // Further additions
    jetprops->Branch("ptd", &ptd, "ptd/F");
    jetprops->Branch("jetcharge03", &jetcharge03, "jetcharge03/F");
    jetprops->Branch("jetcharge05", &jetcharge05, "jetcharge05/F");
    jetprops->Branch("jetcharge07", &jetcharge07, "jetcharge07/F");
    jetprops->Branch("jetcharge10", &jetcharge10, "jetcharge10/F");
    // ----------------
    // Nsubjettiness
    jetprops->Branch("tau1", &tau1, "tau1/F");
    jetprops->Branch("tau2", &tau2, "tau2/F");
    jetprops->Branch("tau3", &tau3, "tau3/F");
    jetprops->Branch("tau4", &tau4, "tau4/F");
    jetprops->Branch("tau5", &tau5, "tau5/F");
    jetprops->Branch("tau2tau1", &tau2tau1, "tau2tau1/F");
    jetprops->Branch("tau3tau2", &tau3tau2, "tau3tau2/F");
    // ----------------
    // Dynamical Grooming
    jetprops->Branch("kappa_TD", &kappa_TD, "kappa_TD/F");
    jetprops->Branch("kappa_ktD", &kappa_ktD, "kappa_ktD/F");
    jetprops->Branch("kappa_zD", &kappa_zD, "kappa_zD/F");

    jetprops->Branch("zg_TD", &zg_TD, "zg_TD/F");
    jetprops->Branch("zg_ktD", &zg_ktD, "zg_ktD/F");
    jetprops->Branch("zg_zD", &zg_zD, "zg_zD/F");

    jetprops->Branch("deltaR_TD", &deltaR_TD, "deltaR_TD/F");
    jetprops->Branch("deltaR_ktD", &deltaR_ktD, "deltaR_ktD/F");
    jetprops->Branch("deltaR_zD", &deltaR_zD, "deltaR_zD/F");
    // ----------------

    // Including SoftDrop Jet Observables

    Float_t SD_jet_eta = 0, SD_jet_rapidity = 0, SD_jet_phi = 0, SD_jet_pt = 0;
    Float_t SD_mass = 0, SD_mz2 = 0, SD_mr = 0, SD_mr2 = 0, SD_rz = 0, SD_r2z = 0;
    Int_t SD_nconst = 0;

    // Further additions
    Float_t SD_ptd = 0;
    Float_t SD_jetcharge03 = 0;
    Float_t SD_jetcharge05 = 0;
    Float_t SD_jetcharge07 = 0;
    Float_t SD_jetcharge10 = 0;
    // ----------------
    // Nsubjettiness
    Float_t SD_tau1 = 0, SD_tau2 = 0, SD_tau3 = 0, SD_tau4 = 0, SD_tau5 = 0, SD_tau2tau1 = 0, SD_tau3tau2 = 0;
    // ----------------

    jetprops->Branch("SD_pt", &SD_jet_pt, "SD_pt/F");
    jetprops->Branch("SD_eta", &SD_jet_eta, "SD_eta/F");
    jetprops->Branch("SD_rapidity", &SD_jet_rapidity, "SD_rapidity/F");
    jetprops->Branch("SD_phi", &SD_jet_phi, "SD_phi/F");
    jetprops->Branch("SD_nconst", &SD_nconst, "SD_nconst/I");
    jetprops->Branch("SD_mass", &SD_mass, "SD_mass/F");
    jetprops->Branch("SD_mz2", &SD_mz2, "SD_mz2/F");
    jetprops->Branch("SD_mr", &SD_mr, "SD_mr/F");
    jetprops->Branch("SD_mr2", &SD_mr2, "SD_mr2/F");
    jetprops->Branch("SD_rz", &SD_rz, "SD_rz/F");
    jetprops->Branch("SD_r2z", &SD_r2z, "SD_r2z/F");
    // Further additions
    jetprops->Branch("SD_ptd", &SD_ptd, "SD_ptd/F");
    jetprops->Branch("SD_jetcharge03", &SD_jetcharge03, "SD_jetcharge03/F");
    jetprops->Branch("SD_jetcharge05", &SD_jetcharge05, "SD_jetcharge05/F");
    jetprops->Branch("SD_jetcharge07", &SD_jetcharge07, "SD_jetcharge07/F");
    jetprops->Branch("SD_jetcharge10", &SD_jetcharge10, "SD_jetcharge10/F");
    // ----------------
    // Nsubjettiness
    jetprops->Branch("SD_tau1", &SD_tau1, "SD_tau1/F");
    jetprops->Branch("SD_tau2", &SD_tau2, "SD_tau2/F");
    jetprops->Branch("SD_tau3", &SD_tau3, "SD_tau3/F");
    jetprops->Branch("SD_tau4", &SD_tau4, "SD_tau4/F");
    jetprops->Branch("SD_tau5", &SD_tau5, "SD_tau5/F");
    jetprops->Branch("SD_tau2tau1", &SD_tau2tau1, "SD_tau2tau1/F");
    jetprops->Branch("SD_tau3tau2", &SD_tau3tau2, "SD_tau3tau2/F");
    // ----------------
    
    // Jet Substructure tree
    int depth = 0;
    jetprops->Branch("depth", &depth, "depth/I");

    std::vector<float>* z = new std::vector<float>;
    std::vector<float>* delta = new std::vector<float>;
    std::vector<float>* kperp = new std::vector<float>;
    std::vector<float>* minv = new std::vector<float>;

    jetprops->Branch("z", &z);
    jetprops->Branch("delta", &delta);
    jetprops->Branch("kperp", &kperp);
    jetprops->Branch("minv", &minv);


    // ----------------------------------

    // get the first event
    HepMC::GenEvent *evt = ascii_in.read_next_event();
    if (!evt)
        cerr << "Input file not found " << inname << endl;

    // loop until we run out of events
    while (evt)
    {

        // analyze the event
        if (debug)
            cout << "Event " << endl;

        evwt = evt->weights()[0]; // set event weight to fill in tree
        // hNEvent->Fill(0.5, evwt); // count events
        // from example_UsingIterators.cc

        float pt_lead = -1;
        float phi_lead = -100;
        float jetR = 0.4;

        int index = 0;
        std::vector<fastjet::PseudoJet> fsParts; // This is final state
        int indexGhosts = 0;
        std::vector<fastjet::PseudoJet> eventGhosts;
        // ------------------------------------
        // Go through all the particles in event
        for (HepMC::GenEvent::particle_iterator pit = evt->particles_begin();
             pit != evt->particles_end(); ++pit)
        {
            const HepMC::GenParticle *p = *pit;
            if (!p->end_vertex() && p->status() != 2 && (!charged_jets || is_charged(p)))
            {
                if (fabs(p->momentum().eta()) < max_eta_track && p->momentum().perp() > ptcut)
                {
                    if (p->momentum().perp() > pt_lead)
                    {
                        pt_lead = p->momentum().perp();
                        phi_lead = p->momentum().phi();
                    }
                    fastjet::PseudoJet jInp(p->momentum().x(), p->momentum().y(), p->momentum().z(), p->momentum().e()); // need masses for E-scheme
                    if (p->status() == 1)
                    {
                        jInp.set_user_info(new PDGInfo(p->pdg_id()));
                        jInp.set_user_index(index);
                        fsParts.push_back(jInp);
                        index++;
                    }
                    else if (p->status() == 3 && do_bkg)
                    {
                        int ghostid;
                        if (rand() / RAND_MAX < 1. / 3.)
                            ghostid = 211;
                        else if (rand() / RAND_MAX < 2. / 3.)
                            ghostid = -211;
                        else
                            ghostid = 111;
                        jInp.set_user_info(new PDGInfo(ghostid));
                        jInp.set_user_index(indexGhosts);
                        eventGhosts.push_back(jInp);
                        indexGhosts++;
                    }
                }
            }
        }
        if (debug)
            cout << "Event read. No final state particles : " << fsParts.size() << endl;
        if (debug && do_bkg)
            cout << "No final state ghosts : " << eventGhosts.size() << endl;
        // Do jet finding
        // Need R =0.2 and R=0.4 later on...
        fastjet::GhostedAreaSpec ghostSpec(max_eta_track, 1, 0.01);
        fastjet::Strategy strategy = fastjet::Best;
        fastjet::RecombinationScheme recombScheme = fastjet::E_scheme; // need E scheme for jet mass
        fastjet::AreaType areaType = fastjet::active_area_explicit_ghosts;
        fastjet::AreaDefinition areaDef = fastjet::AreaDefinition(areaType, ghostSpec);

        fastjet::RangeDefinition range(-max_eta_jet, max_eta_jet, 0, 2. * fastjet::pi);
        fastjet::JetDefinition jetDefCh(fastjet::antikt_algorithm, jetR, recombScheme, strategy);
        // Get loose jets : for sub
        fastjet::RangeDefinition rangeLoose(-_etaloose, _etaloose, 0, 2. * fastjet::pi);
        fastjet::JetDefinition jetDefLoose(fastjet::antikt_algorithm, _dRloose);
        fastjet::ClusterSequence clustSeqLoose(fsParts, jetDefLoose);
        vector<fastjet::PseudoJet> looseJets = clustSeqLoose.inclusive_jets();

        vector<fastjet::PseudoJet> fjInputs;

        if (do_bkg)
        {

            // Produce original event and ghots
            // vector<fastjet::PseudoJet> origEvent;
            // vector<fastjet::PseudoJet> ghosts;
            vector<fastjet::PseudoJet> newEvent;
            // for (auto jet : looseJets)
            // {
            //     for (auto part : fsParts)
            //     {
            //         if (jet.delta_R(part) < _dRloose)
            //         {
            //             origEvent.push_back(part);
            //         }
            //     }
            //     for (auto ghost : eventGhosts)
            //     {
            //         if (jet.delta_R(ghost) < _dRloose)
            //         {
            //             ghosts.push_back(ghost);
            //         }
            //     }
            // }
            // // Subtract ghosts from event
            // newEvent = ConstSubPartEvent(origEvent, ghosts);
            newEvent = ConstSubPartEvent(fsParts, eventGhosts);
            fjInputs = newEvent;
        }
        else
        {
            fjInputs = fsParts;
        }

        vector<fastjet::PseudoJet> corrected_jets;
        fastjet::ClusterSequenceArea clustSeqCh(fjInputs, jetDefCh, areaDef);
        corrected_jets = clustSeqCh.inclusive_jets();
        // modification 12/05/2014
        jet_eta = 0;
        jet_rapidity = 0;
        jet_phi = 0;
        jet_pt = 0;

        if (debug > 0)
            cout << corrected_jets.size() << " jets found" << endl;

        // Go through each Jet of the event
        for (unsigned int iJet = 0; iJet < corrected_jets.size(); iJet++)
        {
            if (!range.is_in_range(corrected_jets[iJet]))
                continue;

            jet_pt = corrected_jets[iJet].perp();
            jet_eta = corrected_jets[iJet].eta();
            jet_rapidity = corrected_jets[iJet].rapidity();
            jet_phi = corrected_jets[iJet].phi();

            if (jet_pt > min_jet_pt)
            {

                fastjet::PseudoJet &jet = corrected_jets[iJet];
                if (debug)
                    cout << " | jet eta : " << jet_eta << " | jet pt : " << jet_pt << " | jet phi : " << jet_phi << " | n const : " << jet.constituents().size() << endl;
                // float rm, rs, r2m, r2s, zs, rz, r2z;
                float zs;
                if (debug)
                    cout << "starting getmasssingulariites" << endl;
                getmassangularities(jet, mr, mr2, zs, mz2, rz, r2z);
                if (debug)
                    cout << "done with getmasssingulariites" << endl;

                getcharge(jet, jetcharge03, 0.3);
                getcharge(jet, jetcharge05, 0.5);
                getcharge(jet, jetcharge07, 0.7);
                getcharge(jet, jetcharge10, 1.0);

                mass = jet.m();

                // Further additions
                ptd = pTD(jet);
                // ----------------

                // Nsubjetiness
                tau1 = 0.0;
                tau2 = 0.0;
                tau3 = 0.0;
                tau4 = 0.0;
                tau5 = 0.0;
                tau2tau1 = 0;
                tau3tau2 = 0;

                tau1 = nSub1(jet) / (jet_pt * jetR);
                tau2 = nSub2(jet) / (jet_pt * jetR);
                tau3 = nSub3(jet) / (jet_pt * jetR);
                tau4 = nSub4(jet) / (jet_pt * jetR);
                tau5 = nSub5(jet) / (jet_pt * jetR);
                tau2tau1 = tau2 / tau1;
                tau3tau2 = tau3 / tau2;
                // -----------
                // SoftDrop
                fastjet::contrib::SoftDrop sd(beta, zcut, jetR);

                fastjet::contrib::SoftDrop sd_ca(beta, zcut, jetR);
                fastjet::Recluster reclust_ca(fastjet::cambridge_aachen_algorithm, fastjet::JetDefinition::max_allowable_R);
                sd_ca.set_reclustering(true, &reclust_ca);

                fastjet::contrib::SoftDrop sd_kt(beta, zcut, jetR);
                fastjet::Recluster reclust_kt(fastjet::kt_algorithm, fastjet::JetDefinition::max_allowable_R);
                sd_kt.set_reclustering(true, &reclust_kt);
                fastjet::contrib::SoftDrop sd_akt(beta, zcut, jetR);
                fastjet::Recluster reclust_akt(fastjet::antikt_algorithm, fastjet::JetDefinition::max_allowable_R);
                sd_akt.set_reclustering(true, &reclust_akt);
                fastjet::contrib::ModifiedMassDropTagger mMDT(zcut);

                // Dynamical grooming
                dyGroomerJet dg_TD(2);
                dyGroomerJet dg_ktD(1);
                dyGroomerJet dg_zD(0.1);

                if (debug > 1)
                    cout << "Softdrop " << jet.constituents().size() << " constituents " << endl;

                zg = -0.1;
                Rg = -0.1;
                kg = 0.0;
                nSD = 0;

                kappa_TD = -1.;
                kappa_ktD = -1.;
                kappa_zD = -1.;

                SD_tau1 = 0.0;
                SD_tau2 = 0.0;
                SD_tau3 = 0.0;
                SD_tau4 = 0.0;
                SD_tau5 = 0.0;
                SD_tau2tau1 = 0;
                SD_tau3tau2 = 0;
                SD_jet_eta = 0.0;
                SD_jet_rapidity = 0.0;
                SD_jet_phi = 0.0;
                SD_jet_pt = 0.0;
                SD_nconst = 0;
                SD_mass = 0.0;

                //if (jet.has_associated_cluster_sequence())
                if (jet.has_valid_cluster_sequence())
                {
                    if (jet.has_pieces())
                    {
                        // SoftDrop
                        std::vector<fastjet::PseudoJet> real_constits;
                        for (auto &p : jet.constituents()) {
                            if (!p.is_pure_ghost())
                                real_constits.push_back(p);
                        }

                        fastjet::JetDefinition jet_def_ca(fastjet::cambridge_aachen_algorithm, jetR);
                        //remove ghosts for reclustering
                        fastjet::ClusterSequence reclust_seq(real_constits, jet_def_ca);
                        auto reclust_jets = reclust_seq.inclusive_jets();
                        
                        if (reclust_jets.empty()) continue; // segurança
                        
                        fastjet::PseudoJet reclustered_jet = reclust_jets[0];
                        std::vector<fastjet::PseudoJet> no_ghosts;
                        for (auto &p : jet.constituents()) {
                            if (!p.is_pure_ghost())
                                no_ghosts.push_back(p);
                        }
                        
                        fastjet::ClusterSequenceArea reclust_seq_area(no_ghosts, jet_def_ca, areaDef);
                        auto reclust_jets_area = reclust_seq_area.inclusive_jets();
                        
                        fastjet::PseudoJet sd_groomed_jet;
                        
                        if (!reclust_jets_area.empty()) {
                            std::sort(reclust_jets.begin(), reclust_jets.end(), [](const fastjet::PseudoJet &a, const fastjet::PseudoJet &b) {
                                return a.perp() > b.perp();
                            });
                        
                            fastjet::PseudoJet reclustered_jet_area = reclust_jets[0];
                        
                            fastjet::contrib::SoftDrop sd_final(beta, zcut, jetR);
                            sd_final.set_reclustering(false); 
                            sd_groomed_jet = sd_final(reclustered_jet_area);
                        }
                        
                        if (sd_groomed_jet.has_structure_of<fastjet::contrib::SoftDrop>())
                        {
                            zg = sd_groomed_jet.structure_of<fastjet::contrib::SoftDrop>().symmetry(); // or mu() or delta_R()
                            Rg = sd_groomed_jet.structure_of<fastjet::contrib::SoftDrop>().delta_R();  // or mu() or delta_R()

                            // iterate to get substructures
                            fastjet::PseudoJet j1 = sd_groomed_jet, j2;
                            
                            depth = 0;
                            const int max_array = 100;
                            
                            if (sd_groomed_jet.has_parents(j1, j2)) {
                                double pt1 = j1.perp();
                                double pt2 = j2.perp();
                                double pt_sum = pt1 + pt2;
                                kg = std::min(pt1, pt2) * Rg;
                            
                                // --- coleta o primeiro par ---
                                double zval = pt_sum > 0 ? pt2 / pt_sum : 0;
                                double dR = j1.delta_R(j2);
                                double kperp_val = zval * dR * pt_sum;
                            
                                if (zval >= 0.1)
                                    nSD++;
                            
                                if (depth < max_array) {
                                    z->push_back(zval);
                                    delta->push_back(dR);
                                    kperp->push_back(kperp_val);
                                    minv->push_back((j1+j2).m());
                                    depth++;
                                }
                            
                                // --- percorre a árvore ---
                                while (j1.has_parents(j1, j2)) {
                                    if (j1.perp() < j2.perp()) std::swap(j1, j2);
                            
                                    pt1 = j1.perp();
                                    pt2 = j2.perp();
                                    pt_sum = pt1 + pt2;
                            
                                    zval = pt_sum > 0 ? pt2 / pt_sum : 0;
                                    dR = j1.delta_R(j2);
                                    kperp_val = zval * dR * pt_sum;
                            
                                    if (zval >= 0.1)
                                        nSD++;
                            
                                    if (depth < max_array) {
                                        z->push_back(zval);
                                        delta->push_back(dR);
                                        kperp->push_back(kperp_val);
                                        minv->push_back((j1+j2).m());
                                        depth++;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        else
                            cout << "No groomed jet structure for jet with  pt " << jet.perp() << " E " << jet.E() << " eta " << jet.eta() << " :  " << jet.constituents().size() << " constituents; jet.has_structure(): " << jet.has_structure() << endl;
                        
                        if (debug > 2)
                            cout << "z_g " << zg << " nSD " << nSD << endl;

                        // Jet observables over the groomed jet
                        float SD_zs;
                        getmassangularities(sd_groomed_jet, SD_mr, SD_mr2, SD_zs, SD_mz2, SD_rz, SD_r2z);

                        getcharge(sd_groomed_jet, SD_jetcharge03, 0.3);
                        getcharge(sd_groomed_jet, SD_jetcharge05, 0.5);
                        getcharge(sd_groomed_jet, SD_jetcharge07, 0.7);
                        getcharge(sd_groomed_jet, SD_jetcharge10, 1.0);

                        //----
                        SD_ptd = pTD(sd_groomed_jet);
                        SD_jet_eta = sd_groomed_jet.eta();
                        SD_jet_rapidity = sd_groomed_jet.rapidity();
                        SD_jet_phi = sd_groomed_jet.phi();
                        SD_jet_pt = sd_groomed_jet.perp();
                        SD_nconst = sd_groomed_jet.constituents().size();
                        SD_mass = sd_groomed_jet.m();
                        // Nsubjetiness
                        SD_tau1 = nSub1(sd_groomed_jet) / (SD_jet_pt * jetR);
                        SD_tau2 = nSub2(sd_groomed_jet) / (SD_jet_pt * jetR);
                        SD_tau3 = nSub3(sd_groomed_jet) / (SD_jet_pt * jetR);
                        SD_tau4 = nSub4(sd_groomed_jet) / (SD_jet_pt * jetR);
                        SD_tau5 = nSub5(sd_groomed_jet) / (SD_jet_pt * jetR);
                        SD_tau2tau1 = SD_tau2 / SD_tau1;
                        SD_tau3tau2 = SD_tau3 / SD_tau2;
                            
                        // -----------
                        // DynamicalGrooming
                        // TD
                        fastjet::PseudoJet DGTD_groomed_jet = dg_TD.doGrooming(jet);
                        kappa_TD = jet_pt * jetR * dg_TD.getMinKappa();
                        kappa_TD = 1 / kappa_TD;
                        zg_TD = dg_TD.getZg();
                        deltaR_TD = dg_TD.getDR12();

                        // ktD
                        fastjet::PseudoJet DGktD_groomed_jet = dg_ktD.doGrooming(jet);
                        kappa_ktD = jet_pt * jetR * dg_ktD.getMinKappa();
                        kappa_ktD = 1 / kappa_ktD;
                        zg_ktD = dg_ktD.getZg();
                        deltaR_ktD = dg_ktD.getDR12();

                        // zD
                        fastjet::PseudoJet DGzD_groomed_jet = dg_zD.doGrooming(jet);
                        kappa_zD = jet_pt * jetR * dg_zD.getMinKappa();
                        kappa_zD = 1 / kappa_zD;
                        zg_zD = dg_zD.getZg();
                        deltaR_zD = dg_zD.getDR12();
                    }
                    else
                    {
                        if (debug > 2)
                        {
                            cout << "Jet has no pieces" << endl;
                            cout << "zg" << zg << endl;
                            cout << "Rg" << Rg << endl;
                        }
                    }
                }
                else
                    cout << "No substructure stored with jet" << endl;

                ijet = iJet;
                nconst = jet.constituents().size();
                // Add jet props to Tree
                jetprops->Fill();
                z->clear();
                delta->clear();
                kperp->clear();
                minv->clear();
                if (debug > 2)
                    cout << "Wrote event jets observables to TTree" << endl;
            }
        }

        // delete the created event from memory
        delete evt;
        // read the next event
        ascii_in >> evt;
        ievt++;
    }

    fout.Write();

    fout.Close();

    return 0;
}
