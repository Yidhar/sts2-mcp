using System.Collections;
using System.Buffers.Binary;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Multiplayer.Game.PeerInput;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes.TreasureRooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2McpBridge.Scripts;

internal static partial class BridgeGameApi
{
    private static object BuildRewardsPayload(
        NRewardsScreen? rewardsScreen,
        NProceedButton? roomProceedButton,
        NProceedButton? rewardProceedButton,
        NMapScreen? mapScreen,
        IReadOnlyList<NRewardButton> rewardButtons)
    {
        var visible = IsRewardsScreenVisible(
            rewardsScreen,
            roomProceedButton,
            rewardProceedButton,
            mapScreen,
            rewardButtons);
        return new
        {
            visible,
            terminal_proceed_visible = visible &&
                                       rewardProceedButton is not null &&
                                       IsNodeVisible(rewardProceedButton),
            rewards = rewardButtons.Select((button, index) => new
            {
                index,
                reward = BuildRewardButtonPayload(button)
            }).ToArray()
        };
    }

    private static object BuildRewardButtonPayload(NRewardButton button)
    {
        var reward = ResolveRewardFromControlForLivePayload(button);
        var visibleTexts = CollectButtonPayloadTexts(button, 4)
            .Where(static text => !string.IsNullOrWhiteSpace(text))
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        var primaryText = visibleTexts.FirstOrDefault();

        return reward switch
        {
            CardReward cardReward => new
            {
                text = primaryText,
                texts = visibleTexts,
                type = "card",
                reward_type = "card",
                can_skip = cardReward.CanSkip,
                can_reroll = cardReward.CanReroll,
                option_count = SafeCount(cardReward.Cards)
            },
            GoldReward goldReward => new
            {
                text = primaryText,
                texts = visibleTexts,
                type = "gold",
                reward_type = "gold",
                amount = goldReward.Amount
            },
            RelicReward relicReward => new
            {
                text = primaryText,
                texts = visibleTexts,
                type = "relic",
                reward_type = "relic",
                rarity = relicReward.Rarity.ToString()
            },
            PotionReward => new
            {
                text = primaryText,
                texts = visibleTexts,
                type = "potion",
                reward_type = "potion"
            },
            null => new
            {
                text = primaryText,
                texts = visibleTexts,
                type = "unknown",
                reward_type = "unknown",
                missing = true
            },
            _ => new
            {
                text = primaryText,
                texts = visibleTexts,
                type = reward.GetType().Name,
                reward_type = reward.GetType().Name
            }
        };
    }

    private static object BuildRewardPayload(Reward? reward)
    {
        if (reward is null)
        {
            return new
            {
                type = "unknown",
                reward_type = "unknown",
                missing = true
            };
        }

        return reward switch
        {
            CardReward cardReward => new
            {
                type = "card",
                reward_type = "card",
                can_skip = cardReward.CanSkip,
                can_reroll = cardReward.CanReroll,
                option_count = SafeCount(cardReward.Cards)
            },
            GoldReward goldReward => new
            {
                type = "gold",
                reward_type = "gold",
                amount = goldReward.Amount
            },
            RelicReward relicReward => new
            {
                type = "relic",
                reward_type = "relic",
                rarity = relicReward.Rarity.ToString()
            },
            PotionReward potionReward => new
            {
                type = "potion",
                reward_type = "potion"
            },
            _ => new
            {
                type = reward.GetType().Name,
                reward_type = reward.GetType().Name
            }
        };
    }

    private static string DescribeReward(Reward? reward)
    {
        if (reward is null)
        {
            return "Unknown reward";
        }

        return reward switch
        {
            CardReward cardReward => $"Card reward ({SafeCount(cardReward.Cards)} options)",
            GoldReward goldReward => $"{goldReward.Amount} gold",
            RelicReward => "Relic reward",
            PotionReward => "Potion reward",
            _ => reward.GetType().Name
        };
    }

    private static int SafeCount(IEnumerable? values)
    {
        if (values is null)
        {
            return 0;
        }

        if (values is ICollection collection)
        {
            return collection.Count;
        }

        var count = 0;
        foreach (var _ in values)
        {
            count++;
        }

        return count;
    }

    private static object BuildCardRewardSelectionPayload(
        NCardRewardSelectionScreen? cardRewardScreen,
        IReadOnlyList<NCardHolder> cardRewardOptions,
        Node? cardRewardSkipButton)
    {
        var visible = IsCardRewardSelectionVisible(cardRewardScreen, cardRewardOptions);
        var ready = IsCardRewardSelectionReady(cardRewardScreen);
        return new
        {
            visible,
            ready,
            skip_visible = ready &&
                           cardRewardSkipButton is not null &&
                           IsNodeVisible(cardRewardSkipButton) &&
                           IsButtonEnabled(cardRewardSkipButton),
            options = cardRewardOptions.Select((holder, index) => new
            {
                index,
                card = BuildCardPayload(holder.CardModel)
            }).ToArray()
        };
    }

    private static object BuildCardSelectionPayload(
        Node? cardSelectionScreen,
        IReadOnlyList<NCardHolder> cardSelectionOptions,
        Node? cardSelectionConfirmButton,
        Node? cardSelectionCancelButton,
        Node? cardSelectionCloseButton,
        Node? cardSelectionSkipButton)
    {
        var visible = cardSelectionScreen is not null && IsNodeVisible(cardSelectionScreen);
        var prefs = GetHiddenFieldValue(cardSelectionScreen, "_prefs");
        var state = CaptureCardSelectionUiState(cardSelectionScreen);
        var prompt = visible ? TryGetCardSelectionPrompt(cardSelectionScreen) : null;
        var texts = CollectCardSelectionSurfaceTexts(
            cardSelectionScreen,
            prompt,
            cardSelectionConfirmButton,
            cardSelectionCancelButton,
            cardSelectionCloseButton,
            cardSelectionSkipButton);
        var selectedCount = state.SelectedCount;
        var minSelect = state.MinSelect ?? GetHiddenPropertyValue<int>(prefs, "MinSelect");
        var maxSelect = state.MaxSelect ?? GetHiddenPropertyValue<int>(prefs, "MaxSelect");
        var remainingSelect = ResolveRemainingSelectCount(selectedCount, minSelect, maxSelect);
        var confirmVisible = state.ConfirmReady;
        var skipVisible = cardSelectionSkipButton is not null &&
                          IsNodeVisible(cardSelectionSkipButton) &&
                          IsButtonEnabled(cardSelectionSkipButton);

        return new
        {
            visible,
            screen_type = visible ? cardSelectionScreen!.GetType().Name : null,
            prompt,
            texts,
            remaining_select = remainingSelect,
            selected_count = selectedCount,
            min_select = minSelect,
            max_select = maxSelect,
            selection_ready = state.SelectionReady,
            opened_age_ms = state.OpenedAgeMs,
            requires_manual_confirmation = GetHiddenPropertyValue<bool>(prefs, "RequireManualConfirmation"),
            cancelable = GetHiddenPropertyValue<bool>(prefs, "Cancelable"),
            confirm_visible = confirmVisible,
            cancel_visible = cardSelectionCancelButton is not null &&
                             IsNodeVisible(cardSelectionCancelButton) &&
                             IsButtonEnabled(cardSelectionCancelButton),
            close_visible = cardSelectionCloseButton is not null &&
                            IsNodeVisible(cardSelectionCloseButton) &&
                            IsButtonEnabled(cardSelectionCloseButton),
            skip_visible = skipVisible,
            options = cardSelectionOptions.Select((holder, index) =>
            {
                var optionIndex = GetCardSelectionOptionIndex(cardSelectionScreen, holder, index);
                var selectionId = GetCardSelectionOptionSelectionId(cardSelectionScreen, holder, optionIndex);
                return new
                {
                    index = optionIndex,
                    selection_id = selectionId,
                    action_id = selectionId is not null
                        ? $"card_selection:select:{selectionId}"
                        : $"card_selection:select:{optionIndex}",
                    card = BuildCardPayload(holder.CardModel),
                    is_selected = IsCardSelectionCardSelected(cardSelectionScreen, holder.CardModel)
                };
            }).ToArray()
        };
    }

    private static object BuildCharacterSelectionPayload(
        NCharacterSelectScreen? characterSelectScreen,
        IReadOnlyList<NCharacterSelectButton> characterButtons,
        NCharacterSelectButton? selectedCharacterButton,
        NConfirmButton? embarkButton)
    {
        var visible = characterSelectScreen is not null && IsNodeVisible(characterSelectScreen);
        var selectedIndex = -1;

        for (var index = 0; index < characterButtons.Count; index++)
        {
            if (ReferenceEquals(characterButtons[index], selectedCharacterButton))
            {
                selectedIndex = index;
                break;
            }
        }

        return new
        {
            visible,
            can_embark = visible && embarkButton is not null && IsNodeVisible(embarkButton) && IsButtonEnabled(embarkButton),
            selected_index = selectedIndex >= 0 ? (int?)selectedIndex : null,
            selected_character = BuildCharacterPayload(selectedCharacterButton?.Character),
            options = characterButtons.Select((button, index) => new
            {
                index,
                character = BuildCharacterPayload(button.Character),
                is_selected = ReferenceEquals(button, selectedCharacterButton),
                is_locked = button.IsLocked,
                is_random = button.IsRandom,
                remote_selected_player_count = button.RemoteSelectedPlayers.Count
            }).ToArray()
        };
    }

    private static object BuildRunModeSelectionPayload(
        Node? runModeSubmenu,
        Node? runModeStandardButton,
        Node? runModeDailyButton,
        Node? runModeCustomButton,
        NBackButton? runModeBackButton)
    {
        return new
        {
            visible = runModeSubmenu is not null && IsNodeVisible(runModeSubmenu),
            options = new[]
            {
                BuildRunModeButtonPayload(runModeStandardButton, "standard", "Standard"),
                BuildRunModeButtonPayload(runModeDailyButton, "daily", "Daily"),
                BuildRunModeButtonPayload(runModeCustomButton, "custom", "Custom")
            },
            back_button = BuildRunModeButtonPayload(runModeBackButton, "back", "Back")
        };
    }

    private static object BuildRunModeButtonPayload(Node? button, string semanticAction, string fallbackLabel)
    {
        if (button is null || !IsNodeVisible(button))
        {
            return new
            {
                visible = false,
                semantic_action = semanticAction,
                label = fallbackLabel,
                texts = Array.Empty<string>()
            };
        }

        var texts = CollectButtonPayloadTexts(button, 4);
        var text = texts.FirstOrDefault(static candidate => !string.IsNullOrWhiteSpace(candidate)) ?? fallbackLabel;

        return new
        {
            visible = true,
            semantic_action = semanticAction,
            label = fallbackLabel,
            text,
            texts,
            node_type = button.GetType().FullName
        };
    }

    private static object BuildEventOptionsPayload(
        IReadOnlyList<NEventOptionButton> eventOptionButtons,
        NMapScreen? mapScreen,
        NEventRoom? eventRoom,
        Node? hoverTipSet,
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton)
    {
        if (IsInteractiveMapSurface(mapScreen))
        {
            return new
            {
                visible = false,
                visible_glossary_source = (string?)null,
                visible_glossary_texts = Array.Empty<string>(),
                visible_glossary = Array.Empty<object>(),
                options = Array.Empty<object>()
            };
        }

        var (glossarySource, glossaryTexts, glossaryEntries) = CollectVisibleEventGlossaryTexts(
            eventOptionButtons,
            eventRoom,
            hoverTipSet);
        var currentEventModel = GetHiddenFieldValue(eventRoom, "_event") as EventModel;
        var options = new List<object>(eventOptionButtons.Count + crystalSphereCells.Count + 4);
        options.AddRange(eventOptionButtons.Select((button, index) => BuildEventOptionPayload(button, index, currentEventModel)));
        options.AddRange(
            BuildCrystalSphereEventOptionPayloads(
                crystalSphereScreen,
                crystalSphereCells,
                crystalSphereSmallDivinationButton,
                crystalSphereBigDivinationButton,
                crystalSphereProceedButton,
                eventOptionButtons.Count));
        var isCrystalSphereVisible = crystalSphereScreen is not null && IsNodeVisible(crystalSphereScreen);

        return new
        {
            visible = eventOptionButtons.Count > 0 || options.Count > 0 || isCrystalSphereVisible || glossaryTexts.Count > 0,
            event_id = currentEventModel?.Id.ToString(),
            layout_type = currentEventModel?.LayoutType.ToString(),
            is_deterministic = currentEventModel?.IsDeterministic,
            is_shared = currentEventModel?.IsShared,
            is_finished = currentEventModel?.IsFinished,
            description_key = currentEventModel?.Description?.LocEntryKey,
            encounter_id = currentEventModel?.CanonicalEncounter?.Id.ToString(),
            dynamic_vars = currentEventModel is null
                ? Array.Empty<object>()
                : BuildDynamicVarPayloads(currentEventModel.DynamicVars),
            visible_glossary_source = glossarySource,
            visible_glossary_texts = glossaryTexts.ToArray(),
            visible_glossary = BuildVisibleGlossaryPayload(glossaryEntries),
            options = options.ToArray()
        };
    }

    private static object BuildCrystalSpherePayload(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton)
    {
        var visible = crystalSphereScreen is not null && IsNodeVisible(crystalSphereScreen);
        if (!visible)
        {
            return new
            {
                visible = false,
                cells = Array.Empty<object>()
            };
        }

        var minigame = GetCrystalSphereMinigame(crystalSphereScreen);
        var currentTool = GetCrystalSphereToolName(minigame);
        var divinationsLeft = GetCrystalSphereDivinationCount(minigame);
        var isFinished = GetCrystalSphereIsFinished(minigame);
        var canRevealHiddenCells =
            divinationsLeft.GetValueOrDefault() > 0 &&
            !isFinished &&
            !string.IsNullOrWhiteSpace(currentTool) &&
            !string.Equals(currentTool, "none", StringComparison.OrdinalIgnoreCase);
        var divinationsLeftLabel =
            TryGetLocalNodeText(GetHiddenFieldValue(crystalSphereScreen, "_divinationsLeftLabel") as Node);
        var instructionsTitle =
            TryGetLocalNodeText(GetHiddenFieldValue(crystalSphereScreen, "_instructionsTitleLabel") as Node);
        var instructionsDescription =
            TryGetLocalNodeText(GetHiddenFieldValue(crystalSphereScreen, "_instructionsDescriptionLabel") as Node);

        return new
        {
            visible = true,
            divinations_left = divinationsLeft,
            divinations_left_text = divinationsLeftLabel,
            current_tool = currentTool,
            is_finished = isFinished,
            grid_size = BuildCrystalSphereGridPayload(minigame, crystalSphereCells),
            instructions_title = instructionsTitle,
            instructions_description = instructionsDescription,
            small_divination = BuildCrystalSphereDivinationButtonPayload(
                crystalSphereSmallDivinationButton,
                expectedTool: "small",
                currentTool,
                divinationSize: "1x1",
                fallbackLabel: "Small Divination"),
            big_divination = BuildCrystalSphereDivinationButtonPayload(
                crystalSphereBigDivinationButton,
                expectedTool: "big",
                currentTool,
                divinationSize: "3x3",
                fallbackLabel: "Big Divination"),
            proceed = BuildCrystalSphereProceedPayload(crystalSphereProceedButton),
            cells = crystalSphereCells
                .Select(cell => BuildCrystalSphereCellPayload(cell, canRevealHiddenCells))
                .ToArray()
        };
    }

    private static object[] BuildCrystalSphereEventOptionPayloads(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton,
        int startingIndex)
    {
        return GetCrystalSphereEventOptionDescriptors(
                crystalSphereScreen,
                crystalSphereCells,
                crystalSphereSmallDivinationButton,
                crystalSphereBigDivinationButton,
                crystalSphereProceedButton,
                startingIndex)
            .Select(BuildCrystalSphereEventOptionPayload)
            .ToArray();
    }

    private static IReadOnlyList<CrystalSphereEventOptionDescriptor> GetCrystalSphereEventOptionDescriptors(
        NCrystalSphereScreen? crystalSphereScreen,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells,
        NDivinationButton? crystalSphereSmallDivinationButton,
        NDivinationButton? crystalSphereBigDivinationButton,
        NProceedButton? crystalSphereProceedButton,
        int startingIndex)
    {
        var descriptors = new List<CrystalSphereEventOptionDescriptor>();
        if (crystalSphereScreen is null || !IsNodeVisible(crystalSphereScreen))
        {
            return descriptors;
        }

        var minigame = GetCrystalSphereMinigame(crystalSphereScreen);
        var currentTool = GetCrystalSphereToolName(minigame);
        var divinationsLeft = GetCrystalSphereDivinationCount(minigame).GetValueOrDefault();
        var isFinished = GetCrystalSphereIsFinished(minigame);
        var canRevealHiddenCells =
            divinationsLeft > 0 &&
            !isFinished &&
            !string.IsNullOrWhiteSpace(currentTool) &&
            !string.Equals(currentTool, "none", StringComparison.OrdinalIgnoreCase);

        void AddDescriptor(CrystalSphereEventOptionDescriptor descriptor)
        {
            descriptors.Add(descriptor with { Index = startingIndex + descriptors.Count });
        }

        var smallVisible = crystalSphereSmallDivinationButton is not null && IsNodeVisible(crystalSphereSmallDivinationButton);
        var smallEnabled = smallVisible && IsButtonEnabled(crystalSphereSmallDivinationButton);
        var smallSelected = string.Equals(currentTool, "small", StringComparison.OrdinalIgnoreCase);
        if (smallVisible)
        {
            AddDescriptor(new CrystalSphereEventOptionDescriptor
            {
                Index = 0,
                OptionType = "crystal_sphere_small_divination",
                OptionId = "crystal_sphere:small_divination",
                Title = GetCrystalSphereControlLabel(crystalSphereSmallDivinationButton, "Small Divination"),
                Description = "Select 1x1 divination.",
                DivinationSize = "1x1",
                IsSelected = smallSelected,
                IsEnabled = smallEnabled,
                ActionAvailable = smallEnabled && !smallSelected,
                Execute = () => InvokeCrystalSphereDivinationAction(
                    crystalSphereScreen,
                    crystalSphereSmallDivinationButton!,
                    useBigDivination: false)
            });
        }

        var bigVisible = crystalSphereBigDivinationButton is not null && IsNodeVisible(crystalSphereBigDivinationButton);
        var bigEnabled = bigVisible && IsButtonEnabled(crystalSphereBigDivinationButton);
        var bigSelected = string.Equals(currentTool, "big", StringComparison.OrdinalIgnoreCase);
        if (bigVisible)
        {
            AddDescriptor(new CrystalSphereEventOptionDescriptor
            {
                Index = 0,
                OptionType = "crystal_sphere_big_divination",
                OptionId = "crystal_sphere:big_divination",
                Title = GetCrystalSphereControlLabel(crystalSphereBigDivinationButton, "Big Divination"),
                Description = "Select 3x3 divination.",
                DivinationSize = "3x3",
                IsSelected = bigSelected,
                IsEnabled = bigEnabled,
                ActionAvailable = bigEnabled && !bigSelected,
                Execute = () => InvokeCrystalSphereDivinationAction(
                    crystalSphereScreen,
                    crystalSphereBigDivinationButton!,
                    useBigDivination: true)
            });
        }

        if (canRevealHiddenCells)
        {
            foreach (var cell in crystalSphereCells)
            {
                var entity = cell.Entity;
                if (entity is null || !entity.IsHidden)
                {
                    continue;
                }

                var x = entity.X;
                var y = entity.Y;
                AddDescriptor(new CrystalSphereEventOptionDescriptor
                {
                    Index = 0,
                    OptionType = "crystal_sphere_cell",
                    OptionId = $"crystal_sphere:cell:{x},{y}",
                    Title = $"Reveal cell ({x}, {y})",
                    Description = "Use the selected divination on this cell.",
                    X = x,
                    Y = y,
                    IsHighlighted = entity.IsHighlighted,
                    IsEnabled = true,
                    ActionAvailable = true,
                    Execute = () => InvokeCrystalSphereCellAction(crystalSphereScreen, cell)
                });
            }
        }

        var proceedVisible = crystalSphereProceedButton is not null && IsNodeVisible(crystalSphereProceedButton);
        var proceedEnabled = proceedVisible && IsButtonEnabled(crystalSphereProceedButton);
        if (proceedVisible)
        {
            AddDescriptor(new CrystalSphereEventOptionDescriptor
            {
                Index = 0,
                OptionType = "crystal_sphere_proceed",
                OptionId = "crystal_sphere:proceed",
                Title = GetCrystalSphereControlLabel(crystalSphereProceedButton, "Proceed"),
                IsProceed = true,
                IsEnabled = proceedEnabled,
                ActionAvailable = proceedEnabled,
                Execute = () => InvokeCrystalSphereProceedAction(
                    crystalSphereScreen,
                    crystalSphereProceedButton!)
            });
        }

        return descriptors;
    }

    private static object BuildCrystalSphereEventOptionPayload(CrystalSphereEventOptionDescriptor descriptor)
    {
        return new
        {
            index = descriptor.Index,
            option_type = descriptor.OptionType,
            option_id = descriptor.OptionId,
            title = descriptor.Title,
            description = descriptor.Description,
            is_locked = !descriptor.IsEnabled,
            is_proceed = descriptor.IsProceed,
            is_selected = descriptor.IsSelected,
            is_enabled = descriptor.IsEnabled,
            action_available = descriptor.ActionAvailable,
            divination_size = descriptor.DivinationSize,
            coord = BuildCrystalSphereCoordPayload(descriptor.X, descriptor.Y),
            is_highlighted = descriptor.IsHighlighted
        };
    }

    private static void AddCrystalSphereEventActions(
        List<BridgeResolvedAction> actions,
        BridgeWorldContext context,
        int startingIndex)
    {
        foreach (var descriptor in GetCrystalSphereEventOptionDescriptors(
                     context.CrystalSphereScreen,
                     context.CrystalSphereCells,
                     context.CrystalSphereSmallDivinationButton,
                     context.CrystalSphereBigDivinationButton,
                     context.CrystalSphereProceedButton,
                     startingIndex))
        {
            if (!descriptor.ActionAvailable || descriptor.Execute is null)
            {
                continue;
            }

            var actionId = $"event_option:{descriptor.Index}";
            actions.Add(new BridgeResolvedAction
            {
                ActionId = actionId,
                Payload = new
                {
                    action_id = actionId,
                    kind = "event_option",
                    index = descriptor.Index,
                    label = $"Choose option {descriptor.Index}: {descriptor.Title}",
                    option = BuildCrystalSphereEventOptionPayload(descriptor),
                    screen = context.Screen
                },
                Execute = descriptor.Execute
            });
        }
    }

    private static object BuildCrystalSphereDivinationButtonPayload(
        Node? button,
        string expectedTool,
        string? currentTool,
        string divinationSize,
        string fallbackLabel)
    {
        var visible = button is not null && IsNodeVisible(button);
        var enabled = visible && IsButtonEnabled(button);
        var isSelected = string.Equals(currentTool, expectedTool, StringComparison.OrdinalIgnoreCase);

        return new
        {
            visible,
            enabled,
            is_selected = isSelected,
            action_available = enabled && !isSelected,
            label = GetCrystalSphereControlLabel(button, fallbackLabel),
            divination_size = divinationSize
        };
    }

    private static object BuildCrystalSphereProceedPayload(Node? button)
    {
        var visible = button is not null && IsNodeVisible(button);
        var enabled = visible && IsButtonEnabled(button);
        return new
        {
            visible,
            enabled,
            action_available = enabled,
            label = GetCrystalSphereControlLabel(button, "Proceed")
        };
    }

    private static object BuildCrystalSphereCellPayload(NCrystalSphereCell cell, bool canRevealHiddenCells)
    {
        var entity = cell.Entity;
        var isHidden = entity?.IsHidden ?? true;
        return new
        {
            x = entity?.X,
            y = entity?.Y,
            is_hidden = isHidden,
            is_highlighted = entity?.IsHighlighted ?? false,
            is_hovered = entity?.IsHovered ?? false,
            can_reveal = canRevealHiddenCells && isHidden,
            revealed_item = isHidden
                ? null
                : BuildCrystalSphereRevealedItemPayload(GetHiddenPropertyObjectValue(entity, "Item"))
        };
    }

    private static object? BuildCrystalSphereRevealedItemPayload(object? item)
    {
        if (item is null)
        {
            return null;
        }

        var itemTypeName = item.GetType().Name;
        var itemType = itemTypeName switch
        {
            "CrystalSphereGold" => "gold",
            "CrystalSpherePotion" => "potion",
            "CrystalSphereRelic" => "relic",
            "CrystalSphereCardReward" => "card_reward",
            "CrystalSphereCurse" => "curse",
            _ => NormalizeComparableText(itemTypeName).Replace("-", "_", StringComparison.Ordinal)
        };

        var potionModel = GetHiddenPropertyObjectValue(item, "Potion") as PotionModel ??
                          GetHiddenFieldValue(item, "_potion") as PotionModel;
        var relicModel = GetHiddenPropertyObjectValue(item, "Relic") as RelicModel ??
                         GetHiddenPropertyObjectValue(item, "Model") as RelicModel;

        string? title = itemType switch
        {
            "gold" => "Gold",
            "potion" when potionModel is not null => TryGetTitle(potionModel),
            "relic" when relicModel is not null => TryGetTitle(relicModel),
            "curse" => "Curse",
            _ => null
        };

        var amount = itemType == "gold"
            ? TryGetIntFromPropertyOrField(item, "Amount")
            : null;
        var rarity = itemType == "card_reward"
            ? TextOf(GetHiddenPropertyObjectValue(item, "Rarity") ?? GetHiddenFieldValue(item, "_rarity"))
            : null;

        return new
        {
            item_type = itemType,
            title,
            amount,
            rarity,
            is_good = TryGetBoolFromPropertyOrField(item, "IsGood") ?? false
        };
    }

    private static object? BuildCrystalSphereGridPayload(
        object? minigame,
        IReadOnlyList<NCrystalSphereCell> crystalSphereCells)
    {
        int? width = null;
        int? height = null;

        if (crystalSphereCells.Count > 0)
        {
            width = crystalSphereCells.Max(static cell => cell.Entity?.X ?? -1) + 1;
            height = crystalSphereCells.Max(static cell => cell.Entity?.Y ?? -1) + 1;
        }

        var gridSize = GetHiddenPropertyObjectValue(minigame, "GridSize") ?? GetHiddenFieldValue(minigame, "_gridSize");
        if (gridSize is not null)
        {
            width ??= TryConvertToInt(gridSize) ??
                      TryGetIntFromPropertyOrField(gridSize, "X", "Width", "Columns", "Col");
            height ??= TryConvertToInt(gridSize) ??
                       TryGetIntFromPropertyOrField(gridSize, "Y", "Height", "Rows", "Row");
        }

        if (!width.HasValue && !height.HasValue)
        {
            return null;
        }

        width ??= height;
        height ??= width;
        return new
        {
            width,
            height
        };
    }

    private static object? BuildCrystalSphereCoordPayload(int? x, int? y)
    {
        return x.HasValue && y.HasValue
            ? new
            {
                x,
                y
            }
            : null;
    }

    private static object? GetCrystalSphereMinigame(NCrystalSphereScreen? crystalSphereScreen)
    {
        return GetHiddenFieldValue(crystalSphereScreen, "_entity") ??
               GetHiddenPropertyObjectValue(crystalSphereScreen, "Entity");
    }

    private static int? GetCrystalSphereDivinationCount(object? minigame)
    {
        return TryGetIntFromPropertyOrField(minigame, "DivinationCount", "_divinationCount");
    }

    private static bool GetCrystalSphereIsFinished(object? minigame)
    {
        return TryGetBoolFromPropertyOrField(minigame, "IsFinished", "_isFinished") ?? false;
    }

    private static string? GetCrystalSphereToolName(object? minigame)
    {
        var rawTool = GetHiddenPropertyObjectValue(minigame, "CrystalSphereTool") ??
                      GetHiddenFieldValue(minigame, "_crystalSphereTool") ??
                      GetHiddenFieldValue(minigame, "_currentTool");
        var toolText = rawTool?.ToString();
        if (string.IsNullOrWhiteSpace(toolText))
        {
            return null;
        }

        return toolText.Trim().ToLowerInvariant() switch
        {
            "none" => "none",
            "small" => "small",
            "big" => "big",
            _ => toolText.Trim()
        };
    }

    private static string GetCrystalSphereControlLabel(Node? button, string fallbackLabel)
    {
        var label = TryGetLocalNodeText(button);
        return string.IsNullOrWhiteSpace(label) ? fallbackLabel : label;
    }

    private static object BuildDeckUpgradeSelectionPayload(
        NDeckUpgradeSelectScreen? deckUpgradeScreen,
        IReadOnlyList<NCardHolder> deckUpgradeOptions,
        NConfirmButton? deckUpgradeConfirmButton,
        NBackButton? deckUpgradeCancelButton,
        NBackButton? deckUpgradeCloseButton)
    {
        var visible = deckUpgradeScreen is not null && IsNodeVisible(deckUpgradeScreen);
        var prompt = visible ? TryGetDeckUpgradePrompt(deckUpgradeScreen) : null;
        var texts = CollectDeckUpgradeSurfaceTexts(
            deckUpgradeScreen,
            prompt,
            deckUpgradeConfirmButton,
            deckUpgradeCancelButton,
            deckUpgradeCloseButton);
        var selectedCount = CountSelectedDeckUpgradeCards(deckUpgradeScreen);
        var useSingleSelection = GetHiddenPropertyValue<bool>(deckUpgradeScreen, "UseSingleSelection") ?? false;
        var confirmVisible = deckUpgradeConfirmButton is not null &&
                             IsNodeVisible(deckUpgradeConfirmButton) &&
                             IsButtonEnabled(deckUpgradeConfirmButton);
        return new
        {
            visible = visible,
            use_single_selection = useSingleSelection,
            selected_count = selectedCount,
            prompt,
            texts,
            confirm_visible = confirmVisible,
            cancel_visible = deckUpgradeCancelButton is not null &&
                             IsNodeVisible(deckUpgradeCancelButton) &&
                             IsButtonEnabled(deckUpgradeCancelButton),
            close_visible = deckUpgradeCloseButton is not null &&
                            IsNodeVisible(deckUpgradeCloseButton) &&
                            IsButtonEnabled(deckUpgradeCloseButton),
            options = deckUpgradeOptions.Select((holder, index) => new
            {
                index,
                card = BuildCardPayload(holder.CardModel),
                upgrade_preview = BuildCardUpgradePreviewPayload(holder.CardModel),
                is_selected = IsDeckUpgradeCardSelected(deckUpgradeScreen, holder.CardModel)
            }).ToArray()
        };
    }

    private static object BuildMainMenuPayload(
        Node? mainMenuRoot,
        Node? mainMenuContinueButton,
        IReadOnlyList<Node> mainMenuTextButtons,
        Node? continueRunInfo,
        Node? abandonRunConfirmPopup,
        IReadOnlyList<NPopupYesNoButton> abandonRunConfirmButtons)
    {
        return new
        {
            visible = mainMenuRoot is not null && IsNodeVisible(mainMenuRoot),
            continue_button = BuildMainMenuButtonPayload(mainMenuContinueButton, null, "continue"),
            continue_run_info = BuildContinueRunInfoPayload(continueRunInfo),
            buttons = mainMenuTextButtons
                .Select((button, index) => BuildMainMenuButtonPayload(button, index, TryGetMainMenuSemanticAction(TryGetLocalNodeText(button))))
                .ToArray(),
            abandon_confirm = new
            {
                visible = abandonRunConfirmPopup is not null && IsNodeVisible(abandonRunConfirmPopup),
                buttons = abandonRunConfirmButtons
                    .Select((button, index) => BuildMainMenuButtonPayload(
                        button,
                        index,
                        TryGetAbandonConfirmSemanticAction(TryGetLocalNodeText(button))))
                    .ToArray()
            }
        };
    }

    private static object BuildMainMenuButtonPayload(Node? button, int? index, string? semanticAction)
    {
        if (button is null || !IsNodeVisible(button))
        {
            return new
            {
                visible = false,
                index,
                semantic_action = semanticAction
            };
        }

        var text = TryGetLocalNodeText(button);

        return new
        {
            visible = true,
            index,
            semantic_action = semanticAction,
            text,
            node_type = button.GetType().FullName
        };
    }

    private static object BuildContinueRunInfoPayload(Node? continueRunInfo)
    {
        var visible = continueRunInfo is not null && IsNodeVisible(continueRunInfo);
        if (!visible)
        {
            return new
            {
                visible = false,
                has_result = false,
                texts = Array.Empty<string>()
            };
        }

        var visibleInfo = continueRunInfo!;
        var hasResult = GetHiddenPropertyValue<bool>(visibleInfo, "HasResult") ??
                        (GetHiddenFieldValue(visibleInfo, "<HasResult>k__BackingField") is bool fieldHasResult
                            ? fieldHasResult
                            : (bool?)null) ??
                        false;
        var texts = CollectPromptAndNodeTexts(
            prompt: null,
            maxCount: 8,
            visibleInfo.GetNodeOrNull<Node>("%DateLabel") ??
            GetHiddenFieldValue(visibleInfo, "_dateLabel") as Node,
            visibleInfo.GetNodeOrNull<Node>("%AscensionLabel") ??
            GetHiddenFieldValue(visibleInfo, "_ascensionLabel") as Node,
            visibleInfo.GetNodeOrNull<Node>("%ProgressLabel") ??
            GetHiddenFieldValue(visibleInfo, "_progressLabel") as Node,
            visibleInfo.GetNodeOrNull<Node>("%HealthLabel") ??
            GetHiddenFieldValue(visibleInfo, "_healthLabel") as Node,
            visibleInfo.GetNodeOrNull<Node>("%GoldLabel") ??
            GetHiddenFieldValue(visibleInfo, "_goldLabel") as Node);

        if (texts.Length == 0)
        {
            texts = CollectLocalVisibleText(
                    visibleInfo.GetNodeOrNull<Node>("%ErrorContainer") ??
                    GetHiddenFieldValue(visibleInfo, "_errorContainer") as Node,
                    4,
                    maxDepth: 1)
                .ToArray();
        }

        return new
        {
            visible = true,
            has_result = hasResult,
            texts
        };
    }

}
