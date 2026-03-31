using System.Text.Json.Serialization;

namespace Sts2DraftTracker.Scripts;

/// <summary>One complete run log, serialized to JSON at run end.</summary>
internal sealed class RunLog
{
    [JsonPropertyName("run_id")]
    public string RunId { get; set; } = "";

    [JsonPropertyName("started_at")]
    public string StartedAt { get; set; } = "";

    [JsonPropertyName("ended_at")]
    public string? EndedAt { get; set; }

    [JsonPropertyName("character")]
    public string? Character { get; set; }

    [JsonPropertyName("outcome")]
    public string? Outcome { get; set; } // "victory", "death", "abandon"

    [JsonPropertyName("final_floor")]
    public int FinalFloor { get; set; }

    [JsonPropertyName("final_hp")]
    public int FinalHp { get; set; }

    [JsonPropertyName("final_max_hp")]
    public int FinalMaxHp { get; set; }

    [JsonPropertyName("final_gold")]
    public int FinalGold { get; set; }

    [JsonPropertyName("final_deck")]
    public List<CardInfo> FinalDeck { get; set; } = new();

    [JsonPropertyName("final_relics")]
    public List<string> FinalRelics { get; set; } = new();

    [JsonPropertyName("events")]
    public List<DraftEvent> Events { get; set; } = new();
}

/// <summary>A single draft/build decision during a run.</summary>
internal sealed class DraftEvent
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = ""; // card_reward, card_skip, shop_buy, shop_remove, upgrade, event_choice

    [JsonPropertyName("floor")]
    public int Floor { get; set; }

    [JsonPropertyName("act")]
    public int Act { get; set; }

    [JsonPropertyName("timestamp")]
    public string Timestamp { get; set; } = "";

    [JsonPropertyName("offered")]
    public List<CardInfo>? Offered { get; set; }

    [JsonPropertyName("picked")]
    public CardInfo? Picked { get; set; }

    [JsonPropertyName("skipped")]
    public bool? Skipped { get; set; }

    // Shop-specific
    [JsonPropertyName("gold_before")]
    public int? GoldBefore { get; set; }

    [JsonPropertyName("cost")]
    public int? Cost { get; set; }

    [JsonPropertyName("item_kind")]
    public string? ItemKind { get; set; } // card, relic, potion, removal

    [JsonPropertyName("item_name")]
    public string? ItemName { get; set; }

    [JsonPropertyName("shop_inventory")]
    public ShopSnapshot? ShopInventory { get; set; }

    // Upgrade-specific
    [JsonPropertyName("deck_before")]
    public List<CardInfo>? DeckBefore { get; set; }

    [JsonPropertyName("upgraded_card")]
    public CardInfo? UpgradedCard { get; set; }

    // Event-specific
    [JsonPropertyName("event_name")]
    public string? EventName { get; set; }

    [JsonPropertyName("option_chosen")]
    public string? OptionChosen { get; set; }
}

internal sealed class CardInfo
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("type")]
    public string Type { get; set; } = ""; // Attack, Skill, Power

    [JsonPropertyName("rarity")]
    public string Rarity { get; set; } = "";

    [JsonPropertyName("cost")]
    public int Cost { get; set; }

    [JsonPropertyName("upgraded")]
    public bool Upgraded { get; set; }
}

internal sealed class ShopSnapshot
{
    [JsonPropertyName("gold")]
    public int Gold { get; set; }

    [JsonPropertyName("cards")]
    public List<ShopItem> Cards { get; set; } = new();

    [JsonPropertyName("relics")]
    public List<ShopItem> Relics { get; set; } = new();

    [JsonPropertyName("potions")]
    public List<ShopItem> Potions { get; set; } = new();

    [JsonPropertyName("removal_cost")]
    public int? RemovalCost { get; set; }

    [JsonPropertyName("removal_available")]
    public bool RemovalAvailable { get; set; }
}

internal sealed class ShopItem
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    [JsonPropertyName("cost")]
    public int Cost { get; set; }

    [JsonPropertyName("affordable")]
    public bool Affordable { get; set; }
}
