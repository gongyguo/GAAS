#include <bits/stdc++.h>
using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> dis(0.0, 1.0);

void generate_possible_world(int model, int run_times, vector<vector<uint32_t>> gT, vector<vector<uint32_t>> FgT, vector<vector<int>> &PO, vector<vector<double>> probT, int dataset_No, vector<string> dataset)
{
    for (int i = 0; i < run_times; i++)
    {
        for (unsigned int i = 0; i < PO.size(); ++i)
        {
            PO[i].clear();
        }
        ofstream out_pw;
        if (model == 1) // IC
        {
            out_pw.open("./realization/" + dataset[dataset_No] + "_pw_ic" + to_string(i) + ".txt");
            for (long unsigned int v = 0; v < FgT.size(); v++)
            {
                for (long unsigned int u = 0; u < FgT[v].size(); u++)
                {
                    int node = FgT[v][u];
                    auto rand = dis(gen); 
                    double p = probT[v][u];
                    if (rand < p)
                    {
                        PO[node].push_back(v);
                    }
                }
            }
        }
        else
        {
            out_pw.open("./realization/" + dataset[dataset_No] + "_pw_lt" + to_string(i) + ".txt");
            for (long unsigned int v = 0; v < gT.size(); v++)
            {
                if (gT[v].size() == 0)
                {
                    continue;
                }
                int u = gen() % gT[v].size();
                int node = gT[v][u];
                PO[node].push_back(v);
            }
        }
        for (int i = 0; i < gT.size(); i++)
        {
            for (long unsigned int j = 0; j < PO[i].size(); j++)
            {
                out_pw << i << " " << PO[i][j] << endl;
            }
        }
        out_pw.close();
    }
}

void generate_cost_file(int numV, int dataset_No, vector<vector<uint32_t>> FgT, vector<string> dataset)
{
    ofstream outFile("dataset/" + dataset[dataset_No] + "_cost_001DEG.txt");
    for (int i = 0; i < numV; i++)
    {
        double cost = 0.01 + 0.01 * (FgT[i].size());
        outFile << cost << '\n';
    }
    outFile.close();
    cout << "generate 0.01*deg cost file done" << endl;
}

void load_graph(string graph_file, vector<vector<uint32_t>> &gT, vector<vector<uint32_t>> &FgT,vector<vector<double>> &probT, uint32_t &n, uint32_t &m)
{
    cout << "reading graph" << endl;
    ifstream inFile((graph_file).c_str());
    if (!inFile)
    {
        cout << "cannot open graph file." << endl;
        exit(1);
    }
    inFile.seekg(0, std::ios_base::beg);
    uint32_t u, v;
    inFile >> n >> m;
    gT = vector<vector<uint32_t>>(n, vector<uint32_t>());
    FgT = vector<vector<uint32_t>>(n, vector<uint32_t>());
    probT = vector<vector<double>>(n, vector<double>());
    while (!inFile.eof())
    {
        inFile >> u >> v;
        gT[v].push_back(u);
        FgT[u].push_back(v);
    }
    for (uint32_t i = 0; i < n; i++)
    {
        uint32_t in_deg = gT[i].size();
        for (uint32_t j = 0; j < in_deg; j++)
        {
            probT[i].push_back(double(1.0 / in_deg));
        }
    }
}

int main(int argc, char **argv)
{
    vector<std::string> dataset = {"DBLP_sym","Youtube_sym","flickr","livejournal"};

    uint32_t n = 0, m = 0;
    vector<vector<uint32_t>> gT;
    vector<vector<uint32_t>> FgT;
    vector<vector<double>> probT;
    vector<vector<int>> PO;

    int dataset_No = 4;
    bool model = 1; // IC
    int run_times = 10;
    for (int i = 0; i < argc; i++)
    {
        if (argv[i] == std::string("-dataset_No"))
            dataset_No = std::atoi(argv[i + 1]);
        if (argv[i] == std::string("-model"))
            model = std::atoi(argv[i + 1]);
    }
    load_graph("dataset/" + dataset[dataset_No], gT, FgT, probT, n, m);
    generate_cost_file(n, dataset_No, FgT, dataset); // generate degree-based cost.
    PO = vector<vector<int>>(n, vector<int>());
    generate_possible_world(model, run_times, gT, FgT, PO, probT, dataset_No, dataset); // generate possible world
}