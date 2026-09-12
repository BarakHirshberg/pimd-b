#include "params.h"

#include <sstream>
#include <regex>
#include <format>
#include <filesystem>

Params::Params(const std::string& filename, const int& rank) : reader(filename) {
    printStatus("Initializing the simulation parameters", rank);

    if (reader.ParseError() < 0)
        throw std::invalid_argument(std::format("Unable to read the configuration file {}", filename));

    /****** Simulation params ******/
    sim["dt"] = getQuantity("time", reader.Get(Sections::SIMULATION, "dt", "1.0 femtosecond"));
    sim["threshold"] = reader.GetReal(Sections::SIMULATION, "threshold", 0.1);
    sim["gamma"] = reader.GetReal(Sections::SIMULATION, "gamma", -1.0);

    if (std::get<double>(sim["gamma"]) < 0)
        sim["gamma"] = 1 / (100.0 * std::get<double>(sim["dt"]));

    sim["nchains"] = reader.GetInteger(Sections::SIMULATION, "nchains", 4);

    if (int nchains = std::get<int>(sim["nchains"]); nchains < 1)
        throw std::invalid_argument(std::format("The specified number of Nose-Hoover chains ({}) is less than one!", nchains));

    sim["steps"] = static_cast<long>(
        std::stod(reader.Get(Sections::SIMULATION, "steps", "1e5")));  // Scientific notation
    sim["sfreq"] = reader.GetLong(Sections::SIMULATION, "sfreq", 1000); /// @todo: Add support for scientific notation
    sim["nbeads"] = reader.GetInteger(Sections::SIMULATION, "nbeads", 4);

    if (int nbeads = std::get<int>(sim["nbeads"]); nbeads < 1)
        throw std::invalid_argument(std::format("The specified number of beads ({}) is less than one!", nbeads));

    sim["seed"] = static_cast<unsigned int>(std::stod(reader.Get(Sections::SIMULATION, "seed", "1234")));

    /****** Flags ******/
    // Bosonic or distinguishable simulation?
    bool bosonic = reader.GetBoolean(Sections::SIMULATION, "bosonic", false);
    sim["bosonic"] = bosonic;
    // Fix the center of mass?
    sim["fixcom"] = reader.GetBoolean(Sections::SIMULATION, "fixcom", true);
    // Enable periodic boundary conditions?
    sim["pbc"] = reader.GetBoolean(Sections::SIMULATION, "pbc", false);
    // Rigorous periodic springs: sum over periodic images of every link (JCP 163, 024101 (2025))
    sim["winding_springs"] = reader.GetBoolean(Sections::SIMULATION, "winding_springs", false);
    sim["max_wind"] = reader.GetInteger(Sections::SIMULATION, "max_wind", 1);
    if (std::get<bool>(sim["winding_springs"]) && !std::get<bool>(sim["pbc"]))
        throw std::invalid_argument("winding_springs requires periodic boundary conditions (pbc = true)!");
    if (std::get<int>(sim["max_wind"]) < 1)
        throw std::invalid_argument("max_wind must be at least 1!");
    // Couple thermostat to normal modes?
    bool nmthermostat = reader.GetBoolean(Sections::SIMULATION, "nmthermostat", false);
    sim["nmthermostat"] = nmthermostat;

    // Particle relabelling move (bosonic simulations only; see RelabelMove)
    bool relabel = reader.GetBoolean(Sections::SIMULATION, "relabel", false);
    sim["relabel"] = relabel;
    std::string relabel_mode = reader.GetString(Sections::SIMULATION, "relabel_mode", "metropolis");
    if (relabel_mode != "metropolis" && relabel_mode != "shuffle")
        throw std::invalid_argument(std::format("The relabelling mode ({}) must be 'metropolis' or 'shuffle'!", relabel_mode));
    sim["relabel_mode"] = relabel_mode;
    sim["relabel_freq"] = reader.GetInteger(Sections::SIMULATION, "relabel_freq", 100);
    if (int f = std::get<int>(sim["relabel_freq"]); f < 1)
        throw std::invalid_argument(std::format("The relabelling frequency ({}) must be positive!", f));
    sim["relabel_attempts"] = reader.GetInteger(Sections::SIMULATION, "relabel_attempts", 1);
    if (int a = std::get<int>(sim["relabel_attempts"]); a < 1)
        throw std::invalid_argument(std::format("The number of relabelling attempts ({}) must be positive!", a));
    sim["relabel_seed"] = static_cast<unsigned int>(std::stod(reader.Get(Sections::SIMULATION, "relabel_seed",
        reader.Get(Sections::SIMULATION, "seed", "1234"))));
    if (relabel && !bosonic)
        throw std::invalid_argument("The relabelling move is only meaningful for bosonic simulations (bosonic = true)!");

    // Exchange (segment regrowth) move for bosonic simulations; see ExchangeMove
    bool exchange_move = reader.GetBoolean(Sections::SIMULATION, "exchange_move", false);
    sim["exchange_move"] = exchange_move;
    sim["exchange_freq"] = reader.GetInteger(Sections::SIMULATION, "exchange_freq", 100);
    if (int f = std::get<int>(sim["exchange_freq"]); f < 1)
        throw std::invalid_argument(std::format("The exchange move frequency ({}) must be positive!", f));
    sim["exchange_attempts"] = reader.GetInteger(Sections::SIMULATION, "exchange_attempts", 1);
    if (int a = std::get<int>(sim["exchange_attempts"]); a < 1)
        throw std::invalid_argument(std::format("The number of exchange move attempts ({}) must be positive!", a));
    // Maximum number of regrown beads per particle (0 = half of the beads)
    sim["exchange_segment"] = reader.GetInteger(Sections::SIMULATION, "exchange_segment", 0);
    // Largest block of consecutive particles closed into a cycle by one move (2 = pair swaps only)
    sim["exchange_kmax"] = reader.GetInteger(Sections::SIMULATION, "exchange_kmax", 2);
    if (int kmax = std::get<int>(sim["exchange_kmax"]); kmax < 2)
        throw std::invalid_argument(std::format("exchange_kmax ({}) must be at least 2!", kmax));
    sim["exchange_seed"] = static_cast<unsigned int>(std::stod(reader.Get(Sections::SIMULATION, "exchange_seed",
        reader.Get(Sections::SIMULATION, "seed", "1234"))));
    if (exchange_move && !bosonic)
        throw std::invalid_argument("The exchange move is only meaningful for bosonic simulations (bosonic = true)!");

    // Softened exterior link (expanded ensemble); see SoftLinkMove
    bool soft_link = reader.GetBoolean(Sections::SIMULATION, "soft_link", false);
    sim["soft_link"] = soft_link;
    sim["soft_link_freq"] = reader.GetInteger(Sections::SIMULATION, "soft_link_freq", 10);
    if (int f = std::get<int>(sim["soft_link_freq"]); f < 1)
        throw std::invalid_argument(std::format("soft_link_freq ({}) must be positive!", f));
    sim["soft_link_ladder"] = reader.Get(Sections::SIMULATION, "soft_link_ladder", "1,2,4,8,16,32");
    sim["soft_link_wl_step"] = reader.GetReal(Sections::SIMULATION, "soft_link_wl_step", 0.5);
    sim["soft_link_seed"] = static_cast<unsigned int>(std::stod(reader.Get(Sections::SIMULATION,
        "soft_link_seed", reader.Get(Sections::SIMULATION, "seed", "1234"))));
    if (soft_link && !bosonic)
        throw std::invalid_argument("The softened-link move is only meaningful for bosonic simulations!");

    // Imaginary-time shift move (cyclic renumbering of the beads); see TimeShiftMove
    bool timeshift = reader.GetBoolean(Sections::SIMULATION, "timeshift", false);
    sim["timeshift"] = timeshift;
    sim["timeshift_freq"] = reader.GetInteger(Sections::SIMULATION, "timeshift_freq", 100);
    if (int f = std::get<int>(sim["timeshift_freq"]); f < 1)
        throw std::invalid_argument(std::format("The time-shift frequency ({}) must be positive!", f));
    sim["timeshift_seed"] = static_cast<unsigned int>(std::stod(reader.Get(Sections::SIMULATION, "timeshift_seed",
        reader.Get(Sections::SIMULATION, "seed", "1234"))));
    if (timeshift && !bosonic)
        throw std::invalid_argument("The time-shift move is only meaningful for bosonic simulations (bosonic = true)!");

    std::string init_pos_type, init_pos_specification;

    if (!parseTokenParentheses(reader.Get(Sections::SIMULATION, "initial_position", "random"), init_pos_type,
                               init_pos_specification)) {
        throw std::invalid_argument("The coordinate initialization method format is invalid!");
    }

    allowed_coord_init_methods = { "random", "xyz", "grid" }; /// @todo Add "cell" option

    if (!labelInArray(init_pos_type, allowed_coord_init_methods))
        throw std::invalid_argument(std::format("The specified coordinate initialization method ({}) is not supported!",
                                                init_pos_type));
    
    if (init_pos_type == "xyz") {
        try {
            const int dummy = 0;  // Dummy variable for std::make_format_args lvalue reference shenanigans
            // Try using specification as filename format
            std::string formatted_filename = std::vformat(init_pos_specification, std::make_format_args(dummy));
            // If the formatted string remains unchanged, that means it wasn't a format
            if (formatted_filename != init_pos_specification) {
                // Ensure the value is being saved as the right type
                sim["init_pos_first_index"] = static_cast<int>(!std::filesystem::exists(formatted_filename));
                init_pos_type = "xyz_formatted";
            }
        } catch (const std::format_error&) {
            throw std::invalid_argument(
                    std::format("The filename format ({}) for coordinate initialization is invalid!",
                                init_pos_specification)
                  );
        } catch (...) {
            throw std::runtime_error(
                    std::format("Filename format ({}) for coordinate initialization validation failed",
                                init_pos_specification)
                  );
        }
    }
    
    sim["init_pos_type"] = init_pos_type;

    if (init_pos_type == "xyz") {
        sim["init_pos_xyz_filename"] = init_pos_specification;
    } else if (init_pos_type == "xyz_formatted") {
        sim["init_pos_xyz_filename_format"] = init_pos_specification;
    }

    // Allowed velocity initialization methods:
    // "random": samples from Maxwell-Boltzmann distribution
    // "manual": reads from vel_X.dat files
    // "manual(format)": reads from format(X) files
    std::string init_vel_type, init_vel_specification;

    if (!parseTokenParentheses(reader.Get(Sections::SIMULATION, "initial_velocity", "random"), init_vel_type,
                               init_vel_specification)) {
        throw std::invalid_argument("The velocity initialization method format is invalid!");
    }
    
    allowed_vel_init_methods = { "random", "manual" };

    if (!labelInArray(init_vel_type, allowed_vel_init_methods))
        throw std::invalid_argument(std::format("The specified velocity initialization method ({}) is not supported!",
                                                init_vel_type));
    
    if (init_vel_type == "manual" && init_vel_specification != "") {
        try {
            const int dummy = 0;  // Dummy variable for std::make_format_args lvalue reference shenanigans
            // Try using specification as filename format
            std::string formatted_filename = std::vformat(init_vel_specification, std::make_format_args(dummy));
            // If the formatted string remains unchanged, that means it wasn't a format
            if (formatted_filename != init_vel_specification) {
                // Ensure the value is being saved as the right type
                sim["init_vel_first_index"] = static_cast<int>(!std::filesystem::exists(formatted_filename));
                init_vel_type = "manual_formatted";
            }
        } catch (const std::format_error&) {
            throw std::invalid_argument(
                    std::format("The filename format ({}) for velocity initialization is invalid!",
                                init_vel_specification)
                  );
        } catch (...) {
            throw std::runtime_error(
                    std::format("Filename format ({}) for velocity initialization validation failed",
                                init_vel_specification)
                  );
        }
    }

    sim["init_vel_type"] = init_vel_type;

    if (init_vel_type == "manual_formatted") {
        sim["init_vel_manual_filename_format"] = init_vel_specification;
    }
    
    // Implemented time propagators:
    // "cartesian": regular velocity Verlet algorithm, propagating the plain Cartesian coordinates
    // "normal_modes": a velocity Verlet algorithm that propagates the normal modes
    allowed_propagators = { "cartesian", "normal_modes" };
    std::string propagator_type = reader.GetString(Sections::SIMULATION, "propagator", "cartesian");
    // For bosons, normal-mode propagation integrates the free ring polymer of the IDENTITY permutation
    // exactly and applies the difference between the bosonic exterior forces and the identity exterior
    // springs as an additional force on the first and last beads (scheme used by the LAMMPS
    // implementation of Myung, Hirshberg, Parrinello, PRL 128, 045301 (2022)). It must be enabled
    // explicitly because it has been validated only against Cartesian propagation.
    bool bosonic_nm = reader.GetBoolean(Sections::SIMULATION, "bosonic_normal_modes", false);
    if (bosonic && propagator_type == "normal_modes" && !bosonic_nm) {
        throw std::invalid_argument("Normal-mode propagation for bosons is approximate (identity ring polymer + exchange "
                                    "correction force); set bosonic_normal_modes = true to enable it.");
    }
    sim["propagator_type"] = propagator_type;
    
    if (!labelInArray(propagator_type, allowed_propagators))
        throw std::invalid_argument(std::format("The specified time propagator ({}) is not supported!", propagator_type));

    // Implemented thermostats:
    // "langevin": A langevin themrostat coupled to the Cartesian coordinates
    // "nose_hoover" A single Nose-Hoover chain coupled to the whole system
    // "nose_hoover_np" A unique Nose-Hoover chain coupled to each particle
    // "nose_hoover_np_dim" A unique Nose-Hoover chain coupled to each cartezian coordinate of each particle
    // "none": No thermostat (NVE simulation)
    // "pile": path-integral Langevin equation (Ceriotti et al., JCP 133, 124104 (2010)): Langevin on the
    //         normal modes with friction 2*omega_k for the non-centroid modes and gamma for the centroid
    allowed_thermostats = { "langevin", "pile", "nose_hoover", "nose_hoover_np", "nose_hoover_np_dim", "none" };
    std::string thermostat_type = reader.GetString(Sections::SIMULATION, "thermostat", "error");
    if (thermostat_type == "error") {
        throw std::invalid_argument("Thermostat must be specified!");
    }
    if (thermostat_type == "pile" && !nmthermostat) {
        throw std::invalid_argument("The pile thermostat acts on the normal modes: set nmthermostat = true!");
    }
    if (nmthermostat && thermostat_type == "none") {
        throw std::invalid_argument("nmthermostat cannot be used in nve ensemble!");
    }
    if ((reader.GetInteger(Sections::SIMULATION, "nchains", -1) != -1) && ((thermostat_type == "none") || (thermostat_type == "langevin") || (thermostat_type == "pile"))) {
        throw std::invalid_argument("nchains can only be used with Nose-Hoover thermostats!");
    }
    sim["thermostat_type"] = thermostat_type;
    
    if (!labelInArray(thermostat_type, allowed_thermostats))
        throw std::invalid_argument(std::format("The specified thermostat ({}) is not supported!", propagator_type));
        
    /* System params */
    sys["temperature"] = getQuantity("temperature", reader.Get(Sections::SYSTEM, "temperature", "1.0 kelvin"));
    if (double temp = std::get<double>(sys["temperature"]); temp <= 0.0) {
        throw std::invalid_argument(std::format("The specified temperature ({0:4.3f} kelvin) is unphysical!", temp));
    }

    sys["natoms"] = reader.GetInteger(Sections::SYSTEM, "natoms", 1);
    if (int natoms = std::get<int>(sys["natoms"]); natoms < 1)
        throw std::invalid_argument(std::format("The specified number of particles ({}) is smaller than one!", natoms));

    sys["mass"] = getQuantity("mass", reader.Get(Sections::SYSTEM, "mass", "1.0 dalton"));
    if (double mass = std::get<double>(sys["mass"]); mass <= 0.0)
        throw std::invalid_argument(std::format("The provided mass ({0:4.3f}) is unphysical!", mass));

    sys["size"] = getQuantity("length", reader.Get(Sections::SYSTEM, "size", "1.0 picometer"));
    if (double size = std::get<double>(sys["size"]); size <= 0.0)
        throw std::invalid_argument(std::format("The provided system size ({0:4.3f}) is unphysical!", size));

    allowed_int_potential_names = { "aziz", "free", "harmonic", "dipole" };
    allowed_ext_potential_names = { "free", "harmonic", "double_well", "cosine" };

    /****** Interaction potential ******/

    std::string interaction_name = reader.GetString(Sections::INT_POTENTIAL, "name", "free");

    if (!labelInArray(interaction_name, allowed_int_potential_names))
        throw std::invalid_argument(std::format("The specified interaction potential ({}) is not supported!",
                                                interaction_name));

    interaction_pot["name"] = interaction_name;
    interaction_pot["cutoff"] = getQuantity("length", reader.Get(Sections::INT_POTENTIAL, "cutoff", "-1.0 angstrom"));

    if (interaction_name == "free") {
        // In the special case of free particles, the cutoff distance is set to zero
        interaction_pot["cutoff"] = 0.0;
    } else if (interaction_name == "harmonic") {
        // In atomic units, the angular frequency of the oscillator has the same dimensions as the energy
        interaction_pot["omega"] = getQuantity(
            "energy", reader.Get(Sections::INT_POTENTIAL, "omega", "1.0 millielectronvolt"));
    } else if (interaction_name == "double_well") {
        interaction_pot["strength"] = getQuantity(
            "energy", reader.Get(Sections::INT_POTENTIAL, "strength", "1.0 millielectronvolt"));
        interaction_pot["location"] = getQuantity(
            "length", reader.Get(Sections::INT_POTENTIAL, "location", "1.0 angstrom"));
    } else if (interaction_name == "dipole") {
        interaction_pot["strength"] = reader.GetReal(Sections::INT_POTENTIAL, "strength", 1.0);
    } else if (interaction_name == "aziz") {
        // Optional smooth cap on the repulsive core (see AzizPotential); 0 = the true Aziz potential
        interaction_pot["v_cap"] = getQuantity(
            "energy", reader.Get(Sections::INT_POTENTIAL, "v_cap", "0.0 kelvin"));
        if (std::get<double>(interaction_pot["v_cap"]) < 0.0)
            throw std::invalid_argument("v_cap must not be negative!");
    }

    /****** External potential ******/

    std::string external_name = reader.GetString(Sections::EXT_POTENTIAL, "name", "free");

    if (!labelInArray(external_name, allowed_ext_potential_names))
        throw std::invalid_argument(std::format("The specified external potential ({}) is not supported!",
                                                external_name));

    external_pot["name"] = external_name;

    if (external_name == "harmonic") {
        // In atomic units, the angular frequency of the oscillator has the same dimensions as the energy
        external_pot["omega"] = getQuantity(
            "energy", reader.Get(Sections::EXT_POTENTIAL, "omega", "1.0 millielectronvolt"));
    } else if (external_name == "double_well") {
        external_pot["strength"] = getQuantity(
            "energy", reader.Get(Sections::EXT_POTENTIAL, "strength", "1.0 millielectronvolt"));
        external_pot["location"] = getQuantity(
            "length", reader.Get(Sections::EXT_POTENTIAL, "location", "1.0 angstrom"));
    } else if (external_name == "cosine") {
        external_pot["amplitude"] = getQuantity(
            "energy", reader.Get(Sections::EXT_POTENTIAL, "amplitude", "1.0 millielectronvolt"));
        external_pot["phase"] = reader.GetReal(Sections::EXT_POTENTIAL, "phase", 1.0);
    }

    /****** Output ******/
    states["positions"] = reader.Get(Sections::OUTPUT, "positions", "off");
    states["velocities"] = reader.Get(Sections::OUTPUT, "velocities", "off");
    states["forces"] = reader.Get(Sections::OUTPUT, "forces", "off");
    states["labels"] = reader.Get(Sections::OUTPUT, "labels", "off");

    /****** Observables ******/

    observables["energy"] = reader.Get(Sections::OBSERVABLES, "energy", "kelvin");
    observables["classical"] = reader.Get(Sections::OBSERVABLES, "classical", "off");
    observables["bosonic"] = reader.Get(Sections::OBSERVABLES, "bosonic", "off");
    observables["gsf"] = reader.Get(Sections::OBSERVABLES, "gsf", "off");
    // gsf_extra adds the even-slice potential and the force-squared kinetic correction (PCCP 28, 17846 (2026))
    observables["gsf_extra"] = reader.Get(Sections::OBSERVABLES, "gsf_extra", "off");
    observables["soft_link"] = reader.Get(Sections::OBSERVABLES, "soft_link", "off");
    sim["gsf_alpha"] = reader.GetReal(Sections::OBSERVABLES, "gsf_alpha", 0.0);
    if (double a = std::get<double>(sim["gsf_alpha"]); a < 0.0 || a > 1.0)
        throw std::invalid_argument(std::format("The GSF parameter alpha ({}) must lie in [0, 1]!", a));

    // Exchange diagnostics (bosonic simulations only) and the radial distribution function
    observables["connection"] = reader.Get(Sections::OBSERVABLES, "connection", "off");
    observables["permutation"] = reader.Get(Sections::OBSERVABLES, "permutation", "off");
    observables["winding"] = reader.Get(Sections::OBSERVABLES, "winding", "off");
    observables["rdf"] = reader.Get(Sections::OBSERVABLES, "rdf", "off");
    observables["relabel"] = reader.Get(Sections::OBSERVABLES, "relabel", "off");
    observables["exchange"] = reader.Get(Sections::OBSERVABLES, "exchange", "off");
    observables["timeshift"] = reader.Get(Sections::OBSERVABLES, "timeshift", "off");

    // Settings of the above observables (stored with the simulation parameters)
    sim["n_perm_samples"] = reader.GetInteger(Sections::OBSERVABLES, "n_perm_samples", 1);
    if (int n = std::get<int>(sim["n_perm_samples"]); n < 1)
        throw std::invalid_argument(std::format("The number of permutation samples ({}) must be positive!", n));

    sim["rdf_bins"] = reader.GetInteger(Sections::OBSERVABLES, "rdf_bins", 100);
    if (int n = std::get<int>(sim["rdf_bins"]); n < 1)
        throw std::invalid_argument(std::format("The number of rdf bins ({}) must be positive!", n));

    // A non-positive rdf_rmax means "half the box size"
    sim["rdf_rmax"] = getQuantity("length", reader.Get(Sections::OBSERVABLES, "rdf_rmax", "-1.0 angstrom"));
}

bool Params::labelInArray(const std::string& label, const StringsList& arr) {
    return std::ranges::find(arr, label) != arr.end();
}

// Splits a string into several strings based on a delimiter
std::vector<std::string> Params::splitString(const std::string& line, char delimiter) {
    std::stringstream ss(line);
    std::vector<std::string> tokens;
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        if (!token.empty())
            tokens.push_back(token);
    }

    return tokens;
}

// Parse a string containing a numerical value and a unit
std::pair<double, std::string> Params::parseQuantity(const std::string& input) {
    std::istringstream iss(input);
    std::string unit;

    // Read the floating-point number and the unit
    // See: https://en.cppreference.com/w/cpp/io/basic_istream/operator_gtgt
    if (double value; iss >> value >> std::ws >> unit) {
        return std::make_pair(value, unit);
    }

    throw std::invalid_argument("Invalid input format");
}

/**
 * Accepts a string of the format "<number> <unit>" and returns the numerical
 * value of the quantity in internal (atomic) units.
 *
 * @param family The family of units to which the quantity belongs.
 * @param input The string containing the numerical value and the unit.
 * @return The numerical value of the quantity in internal units.
 */
double Params::getQuantity(const std::string& family, const std::string& input) {
    // Extract numerical value and specified unit
    auto [value, raw_unit] = parseQuantity(input);

    // Match the provided unit to known units.
    return Units::convertToInternal(family, raw_unit, value);
}

// Used to parse strings of the format "token(value)"
bool Params::parseTokenParentheses(const std::string& input, std::string& token, std::string& value) {
    // Define a regular expression for the specified format
    // @todo Generalize regex to catch cases such as "foo()" and "foo" (without parentheses)	
    // Catch "foo(bar)", "foo()" and "foo"
    const std::regex pattern(R"(\s*(\w+)\((.*)\)\s*|\s*(\w+)\s*)");

    // Match the input string against the pattern
    if (std::smatch matches; std::regex_match(input, matches, pattern)) {
        // Check if there are matched groups
        if (matches.size() == 4 && (matches[1].matched || matches[3].matched)) {
            // Extract token and value from the matched groups
            token = (matches[1].matched) ? matches[1].str() : matches[3].str();
            value = matches[2].str();
            return true;
        }
    }

    // If no match is found or the match does not have the expected groups, return false
    return false;
}
